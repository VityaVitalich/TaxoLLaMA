import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm_notebook as tqdm

from torch.utils.data import Dataset

from .prompt_schemas import (
    hypo_term_hyper,
    predict_child_from_2_parents,
    predict_child_from_parent,
    predict_child_with_parent_and_grandparent,
    predict_children_with_parent_and_brothers,
    predict_parent_from_child_granparent,
    predict_parent_from_child,
    predict_multiple_parents_from_child,
)
import pandas as pd
from multiprocessing import cpu_count
from torch.utils.data import DataLoader


class HypernymDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        tokenizer_encode_args={"return_tensors": "pt"},
        semeval_format=False,
        gold_path=None,
        transforms={
            "only_child_leaf": predict_child_with_parent_and_grandparent,  # заменить на предсказание ребенка
            "only_leafs_all": predict_child_from_parent,
            "only_leafs_divided": predict_children_with_parent_and_brothers,
            "leafs_and_no_leafs": predict_child_from_parent,
            "simple_triplet_grandparent": predict_parent_from_child_granparent,
            "simple_triplet_2parent": predict_child_from_2_parents,
            "predict_hypernym": predict_parent_from_child,
            "predict_multiple_hypernyms": predict_multiple_parents_from_child,
        },
        few_shot_text=''
    ):
        self.tokenizer = tokenizer
        # self.transforms = transforms
        self.tokenizer_encode_args = tokenizer_encode_args
        if semeval_format:
            assert gold_path is not None
            train_data_en_data = pd.read_csv(
                data_path, header=None, sep="\t", names=["term", "relation"]
            )
            train_gold_en_data = pd.read_csv(gold_path, header=None, names=["hypernym"])

            self.df = pd.concat([train_data_en_data, train_gold_en_data], axis=1)[
                ["term", "hypernym"]
            ]
        else:
            # self.df = pd.read_csv(
            #     data_path, header=None, sep="\t", names=["term", "hypernym"]
            # )

            self.data = pd.read_pickle(data_path)

        # self.df.index = list(range(len(self.df)))

        self.case2transform = transforms
        self.few_shot_text = few_shot_text

    def __getitem__(self, index):
        # row = self.df.loc[index]
        # term = row["term"]
        # target = ", ".join(row["hypernym"].split("\t"))
        elem = self.data[index]
        case = elem["case"]

        # if not "changed" in elem.keys():
        #     for field in ["children", "parents", "grandparents", "brothers"]:
        #         if field in elem.keys():
        #             elem[field] = HypernymDataset.delete_techniqal(elem[field])
        #             elem["changed"] = True

        processed_term, target = self.case2transform[case](elem)

        system_prompt = """<s>[INST] <<SYS>> You are a helpfull assistant. List all the possible words divided with a coma. Your answer should not include anything except the words divided by a coma<</SYS>>"""
        processed_term = system_prompt + self.few_shot_text + "\n" + processed_term + "[/INST]"

        encoded_term = self.tokenizer.encode(
            processed_term, **self.tokenizer_encode_args
        )
        encoded_target = self.tokenizer.encode(
            target, add_special_tokens=False, **self.tokenizer_encode_args
        )

        input_seq = torch.concat([encoded_term, encoded_target], dim=1)
        labels = input_seq.clone()
        labels[0, : encoded_term.size()[1]] = -100

        return {
            "encoded_term": encoded_term.squeeze(),  
            "encoded_target": encoded_target.squeeze(0),  
            "input_seq": input_seq.squeeze(),
            "labels": labels.squeeze(), 
        }

    def __len__(self):
        return len(self.data)

    @staticmethod
    def delete_techniqal(elem):
        if isinstance(elem, str):
            if ".n." in elem:
                return elem.split(".")[0].replace("_", " ")
            else:
                return elem.replace("_", " ")

        elif isinstance(elem, list):
            new_words = []
            for word in elem:
                new_words.append(HypernymDataset.delete_techniqal(word))
            return new_words



class Collator:
    def __init__(self, pad_token_id, eos_token_id, mask_token_id):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id

    def __call__(self, batch):
        terms = []
        targets = []
        inputs = []
        labels = []

        for elem in batch:
            terms.append(elem["encoded_term"].flip(dims=[0]))
            targets.append(elem["encoded_target"])
            inputs.append(elem["input_seq"])
            labels.append(elem["labels"])

        terms = torch.nn.utils.rnn.pad_sequence(
            terms, batch_first=True, padding_value=self.pad_token_id
        ).flip(dims=[1])
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.eos_token_id
        )
        inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.mask_token_id
        )

        att_mask_inputs = torch.zeros_like(inputs)
        att_mask_inputs[inputs != self.pad_token_id] = 1

        att_mask_terms = torch.zeros_like(terms)
        att_mask_terms[terms != self.pad_token_id] = 1

        return (terms, att_mask_terms, targets, inputs, att_mask_inputs, labels)


def init_data(tokenizer, config, mask_label_token=-100, semeval_format=False, few_shot=None):

    # should be txt file
    if few_shot:
        with open(few_shot, 'r') as f:
            few_shot_text = f.readlines()

        few_shot_text = ''.join(few_shot_text)
    else:
        few_shot_text = ''

    # data
    train_dataset = HypernymDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
        gold_path=config.gold_path,
        semeval_format=semeval_format,
        few_shot_text=few_shot_text
    )
    test_dataset = HypernymDataset(
        data_path=config.test_data_path,
        tokenizer=tokenizer,
        gold_path=config.test_gold_path,
        semeval_format=semeval_format,
        few_shot_text=few_shot_text
    )

    num_workers = cpu_count()

    collator = Collator(
        tokenizer.eos_token_id, tokenizer.eos_token_id, mask_label_token
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )

    return train_dataset, test_dataset, train_loader, val_loader
