import torch
from torch import nn
import numpy as np

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from torch.utils.data import DataLoader


from .config.config import TaskConfig
from .trainer.train_epoch import train_epoch, predict, validate
from .metrics.metrics import Metric
from .dataset.dataset import HypernymDataset, Collator
from .logger.logger import WanDBWriter

# torch.manual_seed(57)
# torch.cuda.manual_seed(57)
# torch.cuda.manual_seed_all(57)
# np.random.seed(57)
# torch.backends.cudnn.deterministic = True


# https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer


class CustomScheduler:
    def __init__(self, model_size, optimizer, warmup, factor=2):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._step = 0

    def rate(self, step):
        return (
            1
            / self.factor
            * (
                self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
            )
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._step += 1
        rate = self.rate(self._step)
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()


def train(
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    logger,
    config,
    loaded_batch=None,
):
    val_batch = next(iter(val_loader))
    for epoch in range(config.n_epochs):
        print(f"Start of the epoch {epoch}")

        train_epoch(
            model,
            tokenizer,
            optimizer,
            scheduler,
            train_loader,
            val_batch,
            criterion,
            logger,
            config,
            epoch,
            loaded_batch,
        )
        loaded_batch = None

        if (epoch + 1) % config.validation == 0:
            validate(model, val_loader, logger, config)
            print("validated")
        if (epoch + 1) % config.compute_metrics_every == 0:
            all_preds, all_labels = predict(
                model, tokenizer, val_loader, config, epoch=epoch
            )

            metric_calculator = Metric(all_labels, all_preds)
            metrics = metric_calculator.get_metrics()
            print(metrics)
            for key in metrics:
                logger.add_scalar(key, float(metrics[key]))

        if (epoch + 1) % config.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    # 'opt': optimizer.state_dict(),
                    # "sch": scheduler.state_dict(),
                },
                f"{config.saving_path}_epoch={epoch}_MAP={metrics['MAP']}.pth",
            )
            print("saved")


if __name__ == "__main__":
    # create config
    config = TaskConfig()

    # model
    model = AutoModelForCausalLM.from_pretrained(config.model_checkpoint).to(
        config.device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_checkpoint,
        padding_side="left",
    )

    # data
    train_dataset = HypernymDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
        gold_path=config.gold_path,
        semeval_format=True,
    )
    test_dataset = HypernymDataset(
        data_path=config.test_data_path,
        tokenizer=tokenizer,
        gold_path=config.test_gold_path,
        semeval_format=True,
    )

    collator = Collator(tokenizer.eos_token_id, tokenizer.eos_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )
    # optmizations
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = CustomScheduler(config.emb_dim, optimizer, config.warmup)

    # wandb
    logger = WanDBWriter(config)

    # training
    if config.mode == "train":
        train(
            model,
            tokenizer,
            train_loader,
            val_loader,
            scheduler,
            criterion,
            logger,
            config,
        )
    else:
        print("Unknown mode")
