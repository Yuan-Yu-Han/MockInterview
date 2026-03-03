# Finetune — MockInterview 微调流水线

基于 **SFT → DPO** 两阶段流程，对面试 Agent（InterviewerAgent / IntervieweeAgent）进行 LoRA 微调，底座模型为 Qwen3-8B。

---

## 目录结构

```
finetune/
├── config.py          — 微调配置（路径、LoRA 参数、训练超参）
├── generate_data.py   — 批量生成 SFT / DPO 训练数据
├── data_builder.py    — 从报告 JSON 构建 HuggingFace Dataset
├── train_sft.py       — SFT 监督微调训练脚本
└── train_dpo.py       — DPO 直接偏好优化训练脚本
```

---

## 整体流程

```
1. 批量跑面试，生成报告 JSON        (generate_data.py)
         ↓
2. [可选] 生成 DPO 偏好对           (generate_data.py --dpo)
         ↓
3. SFT 监督微调                     (train_sft.py)
         ↓
4. DPO 偏好优化                     (train_dpo.py)
```

---

## 第一步：生成训练数据

### 仅生成 SFT 数据（面试报告）

```bash
python -m finetune.generate_data \
    --resume data/resume/sample_resume.txt \
    --jd     data/documents/sample_jd.txt \
    --sessions 5
```

输出：`data/reports/report_*.json`

### 同时生成 DPO 偏好对

```bash
python -m finetune.generate_data \
    --resume data/resume/sample_resume.txt \
    --jd     data/documents/sample_jd.txt \
    --sessions 5 --dpo
```

额外输出：
- `data/finetune/dpo_interviewee_*.json` — 被面试者偏好对
- `data/finetune/dpo_interviewer_*.json` — 面试官偏好对

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sessions` | 3 | 模拟面试场数 |
| `--questions` | 5 | 每场问题数 |
| `--follow-ups` | 1 | 每题追问数 |
| `--dpo` | 关闭 | 开启后额外生成偏好对 |
| `--chosen-min` | 7.0 | chosen 回答最低分（0-10） |
| `--rejected-max` | 4.0 | rejected 回答最高分（0-10） |

### DPO 偏好对生成逻辑

**Interviewee（回答质量）**：用绝对分数双边验证
- chosen：EvalAgent 在面试中打分 ≥ `chosen-min` 的原始高质量回答
- rejected：模型故意生成的差回答，再由 EvalAgent 打分确认 ≤ `rejected-max`

**Interviewer（问题质量）**：用 pairwise 比较
- 生成一个差问题，再与原始问题做 pairwise 对比，仅当原始问题被选为更好时才写入
- 问题较短、长度相近，pairwise 比绝对打分更可靠

---

## 第二步：SFT 监督微调

```bash
# 微调被面试者 Agent
python -m finetune.train_sft --agent interviewee

# 微调面试官 Agent
python -m finetune.train_sft --agent interviewer
```

### 可选参数

```bash
python -m finetune.train_sft --agent interviewee \
    --data-dir   data/reports \
    --output-dir data/models/sft_interviewee \
    --min-score  7.5 \
    --epochs     5
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent` | 必填 | `interviewee` 或 `interviewer` |
| `--data-dir` | `data/reports` | 报告 JSON 目录 |
| `--output-dir` | `data/models/sft_{agent}` | 模型输出目录 |
| `--model` | config 中路径 | 基础模型路径 |
| `--min-score` | 7.0 | interviewee 正样本最低分 |
| `--epochs` | 3 | 训练轮数 |

**输出**：
- `data/models/sft_{agent}/final/` — 合并后的 LoRA 模型权重
- `data/models/sft_{agent}/` — 各 epoch checkpoint

---

## 第三步：DPO 偏好优化

```bash
# 在 SFT 模型基础上做 DPO（推荐）
python -m finetune.train_dpo --agent interviewee \
    --sft-model data/models/sft_interviewee/final

python -m finetune.train_dpo --agent interviewer \
    --sft-model data/models/sft_interviewer/final
```

若不指定 `--sft-model`，脚本会自动查找 `data/models/sft_{agent}/final/`；找不到则回退到基础模型。

### 可选参数

```bash
python -m finetune.train_dpo --agent interviewee \
    --finetune-dir data/finetune \
    --output-dir   data/models/dpo_interviewee \
    --beta 0.05 \
    --epochs 2
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent` | 必填 | `interviewee` 或 `interviewer` |
| `--sft-model` | 自动查找 | SFT 模型路径 |
| `--finetune-dir` | `data/finetune` | DPO 偏好对目录 |
| `--output-dir` | `data/models/dpo_{agent}` | 模型输出目录 |
| `--beta` | 0.1 | DPO KL 惩罚系数 |
| `--epochs` | 3 | 训练轮数 |

**输出**：`data/models/dpo_{agent}/final/`

---

## 配置说明（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_model` | `/projects/yuan0165/Qwen3-8B` | 底座模型路径 |
| `lora_r` | 16 | LoRA 秩 |
| `lora_alpha` | 32 | LoRA 缩放因子 |
| `lora_dropout` | 0.05 | LoRA dropout |
| `target_modules` | q/k/v/o/gate/up/down_proj | LoRA 注入层 |
| `num_epochs` | 3 | 默认训练轮数 |
| `learning_rate` | 2e-4 | 学习率 |
| `max_seq_length` | 1024 | 最大序列长度 |
| `lr_scheduler_type` | cosine | 学习率调度 |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `dpo_beta` | 0.1 | DPO beta |

训练使用 **4-bit NF4 量化**（BitsAndBytes），计算精度为 bfloat16，显存占用较低。

---

## 环境依赖

```bash
module load Miniconda3 && source activate vllm

pip install transformers trl peft bitsandbytes datasets openai
```

vLLM 服务需在后台运行（用于 generate_data.py 的数据生成）：

```bash
bash scripts/run_vllm.sh
```
