import yaml

with open(r"./configs/ppl_est.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    map(str, params_list["CUDA_VISIBLE_DEVICES"])
)

use_def = params_list["USE_DEF"][0]
SAVING_DIR = os.environ.get("SAVING_DIR")
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"

os.environ["USE_DEF_PROMPT"] = str(use_def)
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
    predict_parent_from_child,
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
            "predict_hypernym": predict_parent_from_child,
        },
        few_shot_text=''
    ):
        self.tokenizer = tokenizer
        self.tokenizer_encode_args = tokenizer_encode_args
        self.data = data
        self.case2transform = transforms
        self.few_shot_text=few_shot_text

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


def get_term(s):
    global case

    # print(s)
    term = s.split("|")[-2]
    term = term.split(":")
    if (case == "pred_hypernym") or (case == "leaf_no_leafs"):
        return term[-1].strip()
    elif (case == "simple_triplet_2parent") or (case == "simple_triplet_grandparent"):
        first_parent = term[1].split(" (")[0].strip()
        second_parent = term[2].split("(")[0].strip()

        # in case of grandparent, it would be hyperhypernym_hyponym
        return first_parent + "_" + second_parent


if __name__ == "__main__":
    batch_size = params_list["BATCH_SIZE"][0]
    model_checkpoint = params_list["MODEL_CHECKPOINT"][0]
    out_name = params_list["OUT_NAME"][0]
    in_name = params_list["IN_NAME"][0]
    case = params_list["CASE"][0]
    chkp_time = 100
    torch.manual_seed(params_list["SEED"][0])

    with open(in_name, "rb") as f:
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
    def ppl_over_loader(model, loader, device, term_to_label=None):
        ppl_ls = []
        if not term_to_label:
            term_to_label = {}

        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch
            decoded_terms = tokenizer.batch_decode(terms)
            cur_terms = list(map(get_term, decoded_terms))
            cur_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)

            #  print(cur_terms, cur_targets)
            if (cur_terms[0], cur_targets[0]) in term_to_label.keys():
                continue

            output = model.forward(
                input_seqs.to(device).long(),
                attention_mask=att_mask_input.to(device).long(),
                labels=labels.to(device).long(),
            )

            logits = output["logits"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            ppl = (
                (
                    (loss * (shift_labels != -100)).sum(dim=1)
                    / (shift_labels != -100).sum(dim=1)
                )
                .exp()
                .cpu()
                .tolist()
            )
            ppl_ls.extend(ppl)

            # print(cur_terms)
            # print(cur_targets)
            for cur_ppl, term, target in zip(ppl, cur_terms, cur_targets):
                # print(term, target)
                term_to_label[(term, target)] = cur_ppl

            if (i + 1) % chkp_time == 0:
                with open(out_name, "wb") as f:
                    pickle.dump(term_to_label, f)

        return ppl_ls, term_to_label

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    if params_list["LOAD"][0]:
        with open(out_name, "rb") as f:
            term_to_label = pickle.load(f)
        ppls, term_to_label = ppl_over_loader(
            inference_model, loader, "cuda", term_to_label
        )

    else:
        ppls, term_to_label = ppl_over_loader(inference_model, loader, "cuda")

    with open(out_name, "wb") as f:
        pickle.dump(term_to_label, f)
