# %load_ext autoreload
# %autoreload 2

import os
import yaml

with open(r"./configs/inference.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

use_def_prompt = params_list["USE_DEF_PROMPT"][0]
os.environ["USE_DEF_PROMPT"] = str(use_def_prompt)

use_def_target = params_list["USE_DEF_TARGET"][0]
os.environ["USE_DEF_TARGET"] = str(use_def_target)

use_number_target = params_list["USE_NUMBER_TARGET"][0]
os.environ["USE_NUMBER_TARGET"] = str(use_number_target)

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    map(str, params_list["CUDA_VISIBLE_DEVICES"])
)

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    map(str, params_list["CUDA_VISIBLE_DEVICES"])
)

SAVING_DIR = os.environ.get("SAVING_DIR")
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"
import sys
import torch
import pandas as pd
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import wandb

from pipeline_src.config.config import TaskConfig
from pipeline_src.train import train
from pipeline_src.logger.logger import WanDBWriter
from pipeline_src.trainer.train_epoch import train_epoch, predict
from pipeline_src.dataset.dataset import init_data
from pipeline_src.logger.logger import WanDBWriter
from pipeline_src.metrics.metrics import Metric


if torch.cuda.is_available():
    device = "cuda"
    print("GPU")
else:
    device = "cpu"
    print("CPU")


SEED = params_list["SEED"][0]
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
print(torch.cuda.device_count())

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)


def main(config):
    if config.model_type == "Auto":
        model_type = AutoModelForCausalLM
        tokenizer_type = AutoTokenizer
    elif config.model_type == "Llama":
        model_type = LlamaForCausalLM
        tokenizer_type = LlamaTokenizer

    extra_model_params = {}
    if config.dtype == "half":
        extra_model_params["torch_dtype"] = torch.bfloat16

    if config.qlora == True:
        extra_model_params["load_in_4bit"] = True

    model = model_type.from_pretrained(
        config.model_checkpoint,
        device_map="auto",
        use_auth_token=HF_TOKEN,
        **extra_model_params
    )

    tokenizer = tokenizer_type.from_pretrained(
        config.model_checkpoint,
        padding_side="left",
        use_auth_token=HF_TOKEN,
    )

    if config.qlora == True:
        # model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    if config.using_peft:
        LORA_R = 8
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        LORA_TARGET_MODULES = [
            "q",
            "v",
        ]

        # model = prepare_model_for_int8_training(model)
        config_lora = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            # target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config_lora)
        model.print_trainable_parameters()

    train_dataset, test_dataset, train_loader, val_loader = init_data(tokenizer, config)

    checkpoint = torch.load(config.load_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    torch.cuda.empty_cache()

    prev_predict = None
    all_preds, all_labels = predict(
        model, tokenizer, val_loader, config, ans_load_path=prev_predict
    )

    return all_preds, all_labels


if __name__ == "__main__":
    config = TaskConfig()

    config.n_epochs = params_list["EPOCHS"][0]
    config.batch_size = params_list["BATCH_SIZE"][0]
    config.lr = float(params_list["LR"][0])
    config.min_lr = float(params_list["MIN_LR"][0])

    config.data_path = params_list["DATA_PATH"][0]
    config.gold_path = (
        None  # "SemEval2018-Task9/training/gold/1A.english.training.gold.txt"
    )
    config.test_data_path = params_list["TEST_DATA_PATH"][0]
    config.test_gold_path = (
        None  # "SemEval2018-Task9/test/gold/1A.english.test.gold.txt"
    )

    config.device = device
    config.using_peft = params_list["USING_PEFT"][0]
    config.model_type = params_list["MODEL_TYPE"][0]  # Auto or Llama
    config.wandb_log_dir = SAVING_DIR + "wandb/"
    config.model_checkpoint = params_list["MODEL_CHECKPOINT"][0]
    config.exp_name = (
        config.model_checkpoint.replace("/", "-")
        + params_list["DATA_PREPROC_STYLE"][0]
        + "_"
        + params_list["STRATEGY"][0]
    )

    config.saving_path = (
        SAVING_DIR
        + "model_checkpoints/"
        + config.exp_name
        #  + "_custom_multilang_"
        + params_list["STRATEGY"][0]
    )

    # if params_list["STRATEGY"][0] == "stohastic":
    config.gen_args = {
        "no_repeat_ngram_size": params_list["NO_REPEAT_NGRAM"][0],
        "do_sample": True,
        "min_new_tokens": params_list["MAX_NEW_TOKENS"][0] - 1,
        "max_new_tokens": params_list["MAX_NEW_TOKENS"][0],
        "temperature": params_list["TEMPERATURE"][0],
        "top_k": params_list["TOP_K"][0],
        "num_return_sequences": params_list["NUM_RETURN_SEQUENCES"][0],
        "num_beams": params_list["NUM_BEAMS"][0],
    }
    # elif params_list["STRATEGY"][0] == "beam_search":
    #     config.gen_args = {
    #         "no_repeat_ngram_size": 2,
    #         "num_beams": params_list["NUM_BEAMS"][0],
    #         "early_stopping": True,
    #         "max_new_tokens": params_list["MAX_NEW_TOKENS"][0],
    #         "temperature": params_list["TEMPERATURE"][0],
    #     }

    config.dtype = params_list["DTYPE"][0]
    config.qlora = params_list["QLORA"][0]
    # if params_list["PREV_PREDICT"][0]:
    #     prev_predict = SAVING_DIR + "model_outputs/" + params_list["PREV_PREDICT"][0]
    # else:
    #     prev_predict = None

    config.load_path = SAVING_DIR + "model_checkpoints/" + params_list["LOAD_PATH"][0]

    all_preds, all_labels = main(config)
    metric_calculator = Metric(all_labels, all_preds, "mean")
    metrics = metric_calculator.get_metrics()
    print(metrics)
