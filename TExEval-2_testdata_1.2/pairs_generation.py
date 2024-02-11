import yaml

with open(r"params_tax.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    map(str, params_list["CUDA_VISIBLE_DEVICES"])
)


SAVING_DIR = os.environ.get("SAVING_DIR")
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"


import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
import torch
from tqdm import tqdm
import pickle
import numpy as np


sys.path.append("../pipeline_src/")
from dataset.dataset import HypernymDataset, Collator
from dataset.prompt_schemas import (
    hypo_term_hyper,
    predict_child_from_2_parents,
    predict_child_from_parent,
    predict_child_with_parent_and_grandparent,
    predict_children_with_parent_and_brothers,
    predict_parent_from_child_granparent,
)

from torch.utils.data import DataLoader
from peft import PeftConfig, PeftModel


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
)
import torch


class PplDataset(HypernymDataset):
    def __init__(
        self,
        data,
        tokenizer,
        tokenizer_encode_args={"return_tensors": "pt"},
        transforms={
            "only_child_leaf": predict_parent_from_child_granparent,
            "only_leafs_all": predict_child_from_parent,
            "only_leafs_divided": predict_children_with_parent_and_brothers,
            "leafs_and_no_leafs": predict_child_from_parent,
            "simple_triplet_grandparent": predict_parent_from_child_granparent,
            "simple_triplet_2parent": predict_child_from_2_parents,
        },
    ):
        self.tokenizer = tokenizer
        self.tokenizer_encode_args = tokenizer_encode_args
        self.data = data
        self.case2transform = transforms


class PplEstimator:
    def __init__(self, model, tokenizer, batch_size=4, device="cpu"):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.bs = batch_size
        self.mask_label_token = -100
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def estimate_over_vertex(self, vertex, possible_verteces):
        """
        vertex is a word = str
        possible verteces is a list of possilbe nodes
        """
        if len(possible_verteces) == 0:
            return []
        loader = self.get_loader(vertex, possible_verteces)
        ppls = self.ppl_over_loader(loader)
        return ppls

    @torch.no_grad()
    def ppl_over_loader(self, loader):
        ppl_ls = []

        for batch in loader:
            terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch
            output = self.model.forward(
                input_seqs.to(self.device).long(),
                attention_mask=att_mask_input.to(self.device).long(),
                labels=labels.to(self.device).long(),
            )

            losses = self.loss_fn(
                output["logits"].transpose(1, 2).detach().cpu(), input_seqs
            )
            ppl = (
                (losses * (labels != -100)).sum(dim=1) / (labels != -100).sum(dim=1)
            ).exp()
            ppl_ls.extend(ppl.tolist())

        return ppl_ls

    def get_loader(self, vertex, possible_verteces):
        """
        return DataLoader object
        """

        cased_list = self.get_cased_list(vertex, possible_verteces)
        dataset = PplDataset(cased_list, self.tokenizer)
        collator = Collator(
            self.tokenizer.eos_token_id,
            self.tokenizer.eos_token_id,
            self.mask_label_token,
        )

        return DataLoader(
            dataset,
            batch_size=self.bs,
            collate_fn=collator,
            shuffle=True,
            num_workers=1,
            drop_last=False,
            pin_memory=False,
        )

    def get_cased_list(self, vertex, possible_verteces):
        """
        returns list of cased samples
        """
        ls = []
        for v in possible_verteces:
            elem = {}
            elem["children"] = v
            elem["parents"] = vertex
            elem["grandparents"] = None
            elem["case"] = "leafs_and_no_leafs"

            ls.append(elem)

        return ls


class TaxonomyBuilder:
    def __init__(self, root, all_verteces):
        self.root = root
        self.all_verteces = all_verteces

    def build_taxonomy(self, strategy, **kwargs):
        self.edge_collector = getattr(self, strategy)
        self.collector_params = kwargs

        self.pbar = tqdm(total=34000)
        self.all_edges = []
        self.build_tree(self.root, self.all_verteces)
        self.pbar.close()
        return self.all_edges

    def build_tree(self, root, possible_verteces):
        top_edges_idx = self.edge_collector(
            root, possible_verteces, **self.collector_params
        )
        new_pos_verteces = np.delete(possible_verteces, top_edges_idx)
        for new_edge_idx in top_edges_idx:
            self.all_edges.append((root, possible_verteces[new_edge_idx]))
            self.pbar.update(1)
            self.build_tree(possible_verteces[new_edge_idx], new_pos_verteces)

    @staticmethod
    def ppl_thr_collector(root, possible_verteces, **kwargs):
        ppls = kwargs["ppl_estimator"].estimate_over_vertex(root, possible_verteces)
        return np.where(np.array(ppls) < kwargs["thr"])[0]

    @staticmethod
    def ppl_top_collector(root, possible_verteces, **kwargs):
        ppls = kwargs["ppl_estimator"].estimate_over_vertex(root, possible_verteces)
        return np.argsort(ppls)[: min(kwargs["top_k"], len(ppls))]


if __name__ == "__main__":
    run_name = params_list["NAME"][0]
    batch_size = params_list["BATCH_SIZE"][0]
    strategy = params_list["STRATEGY"][0]
    top_k = params_list["TOP_K"][0]
    model_checkpoint = params_list["MODEL_CHECKPOINT"][0]
    out_name = "gen_" + params_list["OUT_NAME"][0]

    path = "all_nodes_children_gen.pickle"

    with open(path, "rb") as f:
        all_pairs = pickle.load(f)

    config = PeftConfig.from_pretrained(model_checkpoint)
    # Do not forget your token for Llama2 models
    model = LlamaForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=HF_TOKEN,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        config.base_model_name_or_path,
        use_auth_token=HF_TOKEN,
        padding_side="left",
    )
    inference_model = PeftModel.from_pretrained(model, model_checkpoint)

    dataset = PplDataset(all_pairs, tokenizer)
    collator = Collator(tokenizer.eos_token_id, tokenizer.eos_token_id, -100)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=False,
    )

    @torch.no_grad()
    def generate_over_loader(model, loader, device, gen_args):
        term_to_generations = {}

        for batch in tqdm(loader):
            terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch
            output_tokens = model.generate(
                inputs=terms.to(device),
                attention_mask=att_mask_terms.to(device),
                pad_token_id=tokenizer.eos_token_id,
                **gen_args,
            )
            pred_tokens = output_tokens[:, terms.size()[1] :]
            pred_str = tokenizer.batch_decode(
                pred_tokens.cpu(), skip_special_tokens=True
            )

            def get_term(s):
                return s.split("|")[-2].split(":")[-1].strip()

            decoded_terms = tokenizer.batch_decode(terms)
            cur_terms = list(map(get_term, decoded_terms))

            for pred, term in zip(pred_str, cur_terms):
                term_to_generations[term] = pred
                print(pred)
        return term_to_generations

    gen_args = {
        "do_sample": True,
        "num_beams": 8,
        "max_new_tokens": 128,
        "temperature": 0.95,
        "num_return_sequences": 1,
        "top_k": 20,
        "no_repeat_ngram_size": 3,
        "min_new_tokens": 127,
    }

    term_to_generations = generate_over_loader(
        inference_model, loader, "cuda", gen_args
    )

    with open(out_name, "wb") as f:
        pickle.dump(term_to_generations, f)
