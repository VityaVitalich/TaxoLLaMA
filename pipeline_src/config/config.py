from dataclasses import dataclass, field
import torch
from typing import Dict, Any
import os

SAVING_DIR = os.environ.get("SAVING_DIR")


@dataclass
class TaskConfig:
    gen_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": True,
            "num_beams": 1,
            "num_return_sequences": 2,
            "max_new_tokens": 18,
            "top_k": 20,
        }
    )
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    batch_size: int = 64
    dict_size: int = 51

    n_epochs: int = 20
    lr: float = 3e-4
    min_lr: float = 3e-5
    project_name: str = "taxonomy_babelnet"
    show_every: int = 5
    embedding_dim: int = 1024
    save_every: int = 1
    validation: int = 1
    loss_tol: float = 0
    weight_decay: float = 1e-4
    base_factor: int = 48
    exp_name: str = "small_t5_debug"
    compute_metrics_every: int = 1
    full_log: int = 50
    warmup: int = 4000
    model_checkpoint: str = "google/flan-t5-small"
    max_length: int = 100
    block_size: int = 64
    mode: str = "train"
    data_path: str = "./"
    gold_path: str = "./"
    test_data_path: str = "./"
    test_gold_path: str = "./"
    saving_path: str = SAVING_DIR + "model_checkpoint/"
    using_peft: bool = False
    wandb_log_dir: str = "./"
    model_type: str = "Auto"  # Auto or Llama
    saving_predictions_path: str = SAVING_DIR + "model_outputs/"
    log_pred_every: int = 500
    save_every_batch: int = 5000
