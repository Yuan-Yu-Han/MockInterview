"""DPO（直接偏好优化）训练脚本。

用法
────
# 在 SFT 模型基础上做 DPO（推荐）
python -m finetune.train_dpo --agent interviewee \\
    --sft-model data/models/sft_interviewee/final

python -m finetune.train_dpo --agent interviewer \\
    --sft-model data/models/sft_interviewer/final

# 直接在基础模型上做 DPO（跳过 SFT）
python -m finetune.train_dpo --agent interviewee

# 自定义路径
python -m finetune.train_dpo --agent interviewee \\
    --finetune-dir data/finetune \\
    --output-dir   data/models/dpo_interviewee \\
    --beta 0.05 --epochs 2

前提
────
  需先运行 generate_data.py --dpo 生成偏好对文件：
    data/finetune/dpo_interviewee_*.json
    data/finetune/dpo_interviewer_*.json

输出
────
  data/models/dpo_{agent}/final/   — 合并后的 LoRA 模型权重
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from finetune.config import finetune_config as ft_cfg
from finetune.data_builder import build_dpo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO 微调 — 面试 Agent")
    parser.add_argument("--agent",         required=True, choices=["interviewee", "interviewer"],
                        help="微调目标 Agent")
    parser.add_argument("--sft-model",     default=None,
                        help="SFT 模型路径（默认尝试 data/models/sft_{agent}/final）")
    parser.add_argument("--finetune-dir",  default=None,
                        help="DPO 偏好对目录（默认 data/finetune）")
    parser.add_argument("--output-dir",    default=None, help="模型输出目录")
    parser.add_argument("--base-model",    default=None,
                        help="若无 SFT 模型则用此基础模型（默认 config.base_model）")
    parser.add_argument("--beta",          type=float, default=ft_cfg.dpo_beta,
                        help="DPO beta（KL 惩罚系数，默认 0.1）")
    parser.add_argument("--epochs",        type=int,   default=ft_cfg.num_epochs)
    parser.add_argument("--batch-size",    type=int,   default=ft_cfg.per_device_train_batch_size)
    args = parser.parse_args()

    # ── 路径解析 ──────────────────────────────────────────────────────────────
    finetune_dir = (
        Path(args.finetune_dir) if args.finetune_dir else ft_cfg.finetune_data_dir
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else ft_cfg.model_output_dir / f"dpo_{args.agent}"
    )

    # SFT 模型 > 命令行指定基础模型 > config 默认基础模型
    default_sft = ft_cfg.model_output_dir / f"sft_{args.agent}" / "final"
    if args.sft_model:
        model_path = Path(args.sft_model).resolve()
    elif default_sft.exists():
        model_path = default_sft
        print(f"[info] 自动使用 SFT 模型：{model_path}")
    else:
        model_path = Path(args.base_model or ft_cfg.base_model)
        print(f"[info] 未找到 SFT 模型，使用基础模型：{model_path}")

    print(f"Agent        : {args.agent}")
    print(f"模型路径     : {model_path}")
    print(f"偏好对目录   : {finetune_dir}")
    print(f"输出目录     : {output_dir}")
    print(f"DPO beta     : {args.beta}")

    # ── 数据集 ────────────────────────────────────────────────────────────────
    dataset = build_dpo_dataset(args.agent, finetune_dir)
    print(f"训练样本数   : {len(dataset)}")

    # ── 量化配置（4-bit NF4）──────────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 加载 tokenizer 和 model ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO 训练时左填充

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA 配置 ─────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=ft_cfg.lora_r,
        lora_alpha=ft_cfg.lora_alpha,
        lora_dropout=ft_cfg.lora_dropout,
        target_modules=ft_cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── DPO 训练参数 ──────────────────────────────────────────────────────────
    dpo_cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=ft_cfg.gradient_accumulation_steps,
        learning_rate=ft_cfg.learning_rate,
        warmup_ratio=ft_cfg.warmup_ratio,
        lr_scheduler_type=ft_cfg.lr_scheduler_type,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=ft_cfg.max_seq_length,
        report_to="none",
    )

    # ── 训练 ──────────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dataset,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    print("\n开始 DPO 微调...")
    trainer.train()

    # 保存最终模型（合并 LoRA 权重）
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nDPO 微调完成，模型已保存至：{final_dir}")


if __name__ == "__main__":
    main()
