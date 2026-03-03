"""微调流水线配置。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class FinetuneConfig:
    # ── 路径 ──────────────────────────────────────────────────────────────────
    base_model: str = "/projects/yuan0165/Qwen3-8B"

    report_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "reports"
    )
    finetune_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "finetune"
    )
    model_output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "models"
    )

    # ── 分数阈值 ──────────────────────────────────────────────────────────────
    sft_min_score: float = 7.0      # SFT 正样本最低分
    dpo_chosen_min: float = 7.0     # DPO chosen 最低分
    dpo_rejected_max: float = 4.0   # DPO rejected 最高分（备用；当前主要靠生成）

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── 训练超参 ──────────────────────────────────────────────────────────────
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # ── DPO ───────────────────────────────────────────────────────────────────
    dpo_beta: float = 0.1

    # ── vLLM（生成 DPO rejected 样本） ────────────────────────────────────────
    vllm_base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1/")
    )
    vllm_model: str = field(
        default_factory=lambda: os.getenv("VLLM_MODEL_NAME", "Qwen3-8B")
    )
    vllm_api_key: str = "not-needed"


finetune_config = FinetuneConfig()
