"""从面试报告 JSON 构建 SFT / DPO 训练数据集。

SFT 数据集格式
──────────────
  列名：messages
  每条样本是一个 message list（system / user / assistant），
  与 Qwen3 的 chat template 直接兼容。

DPO 数据集格式
──────────────
  列名：prompt, chosen, rejected
  各列均为 message list，遵循 TRL DPOTrainer 的标准输入格式。
  偏好对由 generate_data.py --dpo 生成，经 pairwise 验证后写入文件。
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.prompts import INTERVIEWEE_SYSTEM, INTERVIEWER_SYSTEM


# ─── 内部工具 ──────────────────────────────────────────────────────────────────

def _load_reports(report_dir: Path) -> List[dict]:
    reports = []
    for f in sorted(report_dir.glob("report_*.json")):
        try:
            with open(f, encoding="utf-8") as fp:
                reports.append(json.load(fp))
        except Exception as e:
            print(f"[警告] 跳过 {f.name}：{e}")
    return reports


def _prev_summary(turns: List[dict], idx: int) -> str:
    if idx == 0:
        return ""
    return "\n".join(f"{i+1}. {turns[i]['question']}" for i in range(idx))


def check_diversity(dataset: Dataset, min_distinct_ratio: float = 0.25) -> bool:
    """计算 assistant 回复的 distinct-2 gram 比例，低于阈值时发出警告。

    distinct-2 = 唯一 bigram 数 / 总 bigram 数，衡量输出多样性。
    比例过低说明模型输出开始同质化（mode collapse 预警）。
    """
    all_text = " ".join(
        msg["content"]
        for ex in dataset
        for msg in ex.get("messages", [])
        if msg["role"] == "assistant"
    )
    words = all_text.split()
    if len(words) < 10:
        return True

    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    counts  = Counter(bigrams)
    ratio   = len(counts) / len(bigrams)
    print(f"[diversity] distinct-2: {ratio:.3f}（最低阈值 {min_distinct_ratio}）"
          f"  总 token: {len(words)}")
    if ratio < min_distinct_ratio:
        print(f"[警告] 多样性过低（{ratio:.3f} < {min_distinct_ratio}），"
              "训练集可能存在同质化风险，建议注入外部多样性数据后再训练。")
        return False
    return True


# ─── SFT ───────────────────────────────────────────────────────────────────────

def build_sft_dataset(
    agent: str,
    report_dir: Path,
    min_score: float = 7.0,
    min_content_score: Optional[float] = None,
    min_structure_score: Optional[float] = None,
    min_relevance_score: Optional[float] = None,
    extra_data_dir: Optional[Path] = None,
) -> Dataset:
    """构建 SFT 训练集。

    Args:
        agent: "interviewee" 或 "interviewer"
        report_dir: 含 report_*.json 的目录
        min_score: 综合分最低阈值（0-10）
        min_content_score / min_structure_score / min_relevance_score:
            可选的单维度阈值，与 min_score 取 AND 关系。
            例：min_score=6.5, min_content_score=7 表示"综合分 ≥ 6.5 且内容分 ≥ 7"。
        extra_data_dir: 可选的额外报告目录（用于注入多样性数据，防止 mode collapse）

    Returns:
        HuggingFace Dataset，列名为 "messages"
    """
    if agent not in ("interviewee", "interviewer"):
        raise ValueError(f"agent 须为 interviewee 或 interviewer，得到 {agent!r}")

    dirs = [report_dir]
    if extra_data_dir and extra_data_dir != report_dir:
        dirs.append(extra_data_dir)

    all_reports = []
    for d in dirs:
        all_reports.extend(_load_reports(d))

    if not all_reports:
        raise FileNotFoundError(
            f"在 {report_dir} 下未找到 report_*.json 文件。"
            "请先运行面试并保存报告，或用 generate_data.py 批量生成。"
        )

    examples = []

    for r in all_reports:
        candidate = r.get("candidate_name", "候选人")
        position  = r.get("position", "软件工程师")
        turns     = r.get("detailed_feedback", [])

        if agent == "interviewee":
            for t in turns:
                score    = t.get("score", 0.0)
                question = t.get("question", "").strip()
                answer   = t.get("answer", "").strip()
                if not (question and answer and score >= min_score):
                    continue
                # 单维度额外过滤
                if min_content_score   and t.get("content_score",   0) < min_content_score:
                    continue
                if min_structure_score and t.get("structure_score", 0) < min_structure_score:
                    continue
                if min_relevance_score and t.get("relevance_score", 0) < min_relevance_score:
                    continue

                examples.append({
                    "messages": [
                        {"role": "system",    "content": INTERVIEWEE_SYSTEM},
                        {"role": "user",      "content": f"面试官：{question}"},
                        {"role": "assistant", "content": answer},
                    ]
                })

        else:  # interviewer：所有问题均作为 SFT 正样本，用历史做上文
            for idx, t in enumerate(turns):
                question = t.get("question", "").strip()
                if not question:
                    continue

                if idx == 0:
                    user_msg = (
                        f"候选人：{candidate}，应聘职位：{position}。\n"
                        "请生成第一个面试问题，可以是技术题、行为题或情景题。"
                    )
                else:
                    summary  = _prev_summary(turns, idx)
                    user_msg = (
                        f"候选人：{candidate}，应聘职位：{position}。\n\n"
                        f"已提过的问题：\n{summary}\n\n"
                        "请继续生成下一个面试问题，注意不要重复，且交替使用不同题型。"
                    )

                examples.append({
                    "messages": [
                        {"role": "system",    "content": INTERVIEWER_SYSTEM},
                        {"role": "user",      "content": user_msg},
                        {"role": "assistant", "content": question},
                    ]
                })

    if not examples:
        raise ValueError(
            f"agent={agent!r} 的 SFT 数据集为空。"
            + (f"（min_score={min_score}，请检查报告评分或调低阈值）"
               if agent == "interviewee" else "（报告中无有效问题）")
        )

    dataset = Dataset.from_list(examples)
    print(f"[data_builder] SFT-{agent}：{len(dataset)} 条样本")
    check_diversity(dataset)
    return dataset


# ─── DPO ───────────────────────────────────────────────────────────────────────

def build_dpo_dataset(
    agent: str,
    finetune_data_dir: Path,
    min_confidence: str = "medium",
) -> Dataset:
    """构建 DPO 训练集。

    从 finetune_data_dir 读取 dpo_{agent}_*.json 文件
    （由 generate_data.py --dpo 生成，已含 pairwise 验证结果）。

    Args:
        agent: "interviewee" 或 "interviewer"
        finetune_data_dir: 含 dpo_{agent}_*.json 的目录
        min_confidence: "high" 仅用高置信偏好对；"medium" 含中等置信（默认）

    Returns:
        HuggingFace Dataset，列名为 "prompt"/"chosen"/"rejected"（各为 message list）
    """
    if agent not in ("interviewee", "interviewer"):
        raise ValueError(f"agent 须为 interviewee 或 interviewer，得到 {agent!r}")

    allowed_conf = {"high"} if min_confidence == "high" else {"high", "medium"}

    pattern = f"dpo_{agent}_*.json"
    files = sorted(finetune_data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"在 {finetune_data_dir} 下未找到 {pattern} 文件。\n"
            "请先运行：python -m finetune.generate_data --dpo"
        )

    examples = []
    skipped_conf = 0

    for f in files:
        try:
            pairs = json.loads(f.read_text(encoding="utf-8"))
            for p in pairs:
                sys_msg  = p.get("system", "")
                user_msg = p.get("prompt_user", "")
                chosen   = p.get("chosen", "")
                rejected = p.get("rejected", "")
                if not (user_msg and chosen and rejected):
                    continue

                # 按 pairwise confidence 过滤
                conf = p.get("pairwise", {}).get("confidence", "high")
                if conf not in allowed_conf:
                    skipped_conf += 1
                    continue

                examples.append({
                    "prompt": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                    "chosen":   [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                })
        except Exception as e:
            print(f"[警告] 跳过 {f.name}：{e}")

    if skipped_conf:
        print(f"[data_builder] 因置信度 < {min_confidence} 跳过 {skipped_conf} 条偏好对")

    if not examples:
        raise ValueError(
            f"DPO-{agent} 数据集为空。"
            f"（{finetune_data_dir} 下有数据但全部被置信度过滤？"
            f"尝试 min_confidence='medium'）"
        )

    print(f"[data_builder] DPO-{agent}：{len(examples)} 条偏好对（置信度 ≥ {min_confidence}）")
    return Dataset.from_list(examples)
