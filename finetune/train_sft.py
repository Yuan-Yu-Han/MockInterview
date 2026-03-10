"""SFT（监督微调）训练脚本。

用法
────
# 微调被面试者（IntervieweeAgent）
python -m finetune.train_sft --agent interviewee

# 微调面试官（InterviewerAgent）
python -m finetune.train_sft --agent interviewer

# 自定义路径
python -m finetune.train_sft --agent interviewee \\
    --data-dir  data/reports \\
    --output-dir data/models/sft_interviewee \\
    --min-score 7.5 --epochs 5

输出
────
  data/models/sft_{agent}/final/   — 合并后的 LoRA 模型权重
  data/models/sft_{agent}/         — 各 epoch checkpoint
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from finetune.config import finetune_config as ft_cfg
from finetune.data_builder import build_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 微调 — 面试 Agent")
    parser.add_argument("--agent",      required=True, choices=["interviewee", "interviewer"],
                        help="微调目标 Agent")
    parser.add_argument("--data-dir",   default=None, help="报告 JSON 目录（默认 data/reports）")
    parser.add_argument("--output-dir", default=None, help="模型输出目录")
    parser.add_argument("--model",      default=None, help="基础模型路径（默认 config.base_model）")
    parser.add_argument("--min-score",  type=float, default=ft_cfg.sft_min_score,
                        help="interviewee 正样本最低分（默认 7.0）")
    parser.add_argument("--epochs",     type=int,   default=ft_cfg.num_epochs)
    parser.add_argument("--batch-size", type=int,   default=ft_cfg.per_device_train_batch_size)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)   if args.data_dir   else ft_cfg.report_dir
    output_dir = Path(args.output_dir) if args.output_dir else ft_cfg.model_output_dir / f"sft_{args.agent}"
    base_model = args.model or ft_cfg.base_model

    print(f"Agent      : {args.agent}")
    print(f"数据目录   : {data_dir}")
    print(f"输出目录   : {output_dir}")
    print(f"基础模型   : {base_model}")

    # ── 数据集 ────────────────────────────────────────────────────────────────
    dataset = build_sft_dataset(args.agent, data_dir, min_score=args.min_score)
    print(f"训练样本数 : {len(dataset)}")

    # ── 量化配置（4-bit NF4）──────────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 加载 tokenizer 和 model ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT 训练时右填充

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
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

    # ── 训练参数 ──────────────────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=ft_cfg.gradient_accumulation_steps,
        learning_rate=ft_cfg.learning_rate,
        warmup_ratio=ft_cfg.warmup_ratio,
        lr_scheduler_type=ft_cfg.lr_scheduler_type,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=ft_cfg.max_seq_length,
        report_to="none",
    )

    # ── 训练 ──────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    print("\n开始 SFT 微调...")
    trainer.train()

    # 保存最终模型（合并 LoRA 权重）
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSFT 微调完成，模型已保存至：{final_dir}")


if __name__ == "__main__":
    main()
