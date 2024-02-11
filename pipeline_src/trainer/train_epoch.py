from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
import torch
import gc
import wandb
import json
import itertools
from collections import Counter
import pickle
import pandas as pd
import multiprocessing
import time


class EmptyCacheTimeoutError(Exception):
    pass


def empty_cache_with_timeout(timeout):
    def empty_cache():
        torch.cuda.empty_cache()

    process = multiprocessing.Process(target=empty_cache)

    try:
        process.start()
        process.join(timeout)
    except multiprocessing.TimeoutError:
        process.terminate()
        raise EmptyCacheTimeoutError(
            "torch.cuda.empty_cache() took too long to execute."
        )
    else:
        process.terminate()


def train_epoch(
    model,
    tokenizer,
    optimizer,
    scheduler,
    train_loader,
    val_batch,
    crit,
    logger,
    config,
    epoch,
    loaded_batch=None,
):
    # unfreeze(model)
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, batch in pbar:
        if loaded_batch and batch_idx < loaded_batch:
            continue

        try:
            empty_cache_with_timeout(5)  # Set the timeout (in seconds) as needed
        except EmptyCacheTimeoutError:
            print("Skipping torch.cuda.empty_cache() due to timeout.")

        st = logger.get_step() + 1
        logger.set_step(step=st, mode="train")

        terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch
        output = model.forward(
            input_seqs.to(config.device).long(),
            attention_mask=att_mask_input.to(config.device).long(),
            labels=labels.to(config.device).long(),
        )

        optimizer.zero_grad()
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.add_scalar("loss", loss.item())
        pbar.set_postfix({"Loss": loss.item()})

        if (batch_idx + 1) % config.save_every_batch == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    # 'opt': optimizer.state_dict(),
                    # "sch": scheduler.state_dict(),
                },
                f"{config.saving_path}_epoch={epoch}_batch_idx={batch_idx}.pth",
            )
            previous_checkpoint = (
                f"{config.saving_path}_epoch={epoch}_batch_idx={batch_idx-100}.pth"
            )
        #  if os.path.isfile(previous_checkpoint):
        #     os.remove(previous_checkpoint)
        if (batch_idx + 1) % config.log_pred_every == 0:
            model.eval()
            with torch.no_grad():
                (
                    terms,
                    att_mask_terms,
                    targets,
                    input_seqs,
                    att_mask_input,
                    labels,
                ) = val_batch

                pred, gold = get_one_sample(model, tokenizer, val_batch, config)
                pred_str = [elem[0] for elem in pred]

                question = tokenizer.batch_decode(terms, skip_special_tokens=True)

                df = pd.DataFrame(
                    {"question": question, "predict": pred_str, "gold": gold}
                )
                # print(df)
                logger.wandb.log({"Examples": wandb.Table(dataframe=df)})

            model.train()
    return None
    # return loss ...


@torch.no_grad()
def validate(model, val_loader, logger, config):
    model.eval()

    mean_loss = 0
    for batch_idx, batch in tqdm(enumerate(val_loader)):
        terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch

        with torch.no_grad():
            output = model.forward(
                input_seqs.to(config.device),
                attention_mask=att_mask_input.to(config.device),
                labels=labels.to(config.device),
            )
            loss = output["loss"]
            mean_loss += loss.item()
        torch.cuda.empty_cache()

    mean_loss = mean_loss / (batch_idx + 1)
    logger.add_scalar("Val_loss", mean_loss)

    # del y, batch, output, loss

    return mean_loss


@torch.no_grad()
def predict(model, tokenizer, val_loader, config, epoch="", ans_load_path=None):
    model.eval()

    if ans_load_path:
        with open(ans_load_path, "rb") as fp:
            all_preds = pickle.load(fp)

        assert (
            len(all_preds) % config.batch_size == 0
        ), "preds len and batch does not fit to {}".format(config.batch_size)
    else:
        all_preds = []
    all_labels = []

    saving_path = (
        config.saving_predictions_path + "_" + config.exp_name + "_" + str(epoch)
    )

    evalbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="eval going")
    for batch_idx, batch in evalbar:
        if ans_load_path:
            if batch_idx < (len(all_preds) // config.batch_size):
                continue

        pred, gold = get_one_sample(model, tokenizer, batch, config)

        all_preds.extend(pred)
        all_labels.extend(gold)

        # if batch_idx % 10 == 0:
        # with open(saving_path, "wb") as fp:
        #    pickle.dump(all_preds, fp)

        # print(all_preds)
        torch.cuda.empty_cache()

    with open(saving_path, "wb") as fp:
        pickle.dump(all_preds, fp)
    return all_preds, all_labels


@torch.no_grad()
def get_one_sample(model, tokenizer, batch, config):
    model.eval()

    terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch
    output_tokens = model.generate(
        inputs=terms.to(config.device),
        attention_mask=att_mask_terms.to(config.device),
        pad_token_id=tokenizer.eos_token_id,
        **config.gen_args,
    )
    pred_tokens = output_tokens[:, terms.size()[1] :]
    pred_str = tokenizer.batch_decode(pred_tokens.cpu(), skip_special_tokens=True)
    gold_str = tokenizer.batch_decode(targets, skip_special_tokens=True)

    if len(pred_str) > len(gold_str):
        pred_str = split(pred_str, config.gen_args["num_return_sequences"])

    return pred_str, gold_str


def split(ls, size):
    res = []

    for i in range(0, len(ls) - 1, size):
        res.append(ls[i : i + size])
    return res
