from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    train_file: Path
    output_dir: Path
    model_name_or_path: str = "FacebookAI/xlm-roberta-large"
    text_field: str = "text"
    max_length: int = 256
    eval_ratio: float = 0.1
    seed: int = 42
    num_train_epochs: float = 5.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    logging_steps: int = 20
    save_total_limit: int = 2
    fp16: bool = False
    bf16: bool = False
    auto_clean: bool = True
    negative_ratio: float = 0.0
