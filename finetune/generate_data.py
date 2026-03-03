"""批量生成 SFT / DPO 训练数据。

用法
────
# 仅跑 AI 面试（产生报告 JSON，供 SFT 使用）
python -m finetune.generate_data \\
    --resume data/resume/sample_resume.txt \\
    --jd     data/documents/sample_jd.txt \\
    --sessions 5

# 同时生成 DPO 偏好对
python -m finetune.generate_data \\
    --resume data/resume/sample_resume.txt \\
    --jd     data/documents/sample_jd.txt \\
    --sessions 5 --dpo

DPO 偏好对生成逻辑
──────────────────
Interviewee（回答质量）：
  原始回答已由 EvalAgent 打分 ≥ chosen-min（高分 = chosen）
  → 生成差回答 → EvalAgent 给差回答打分 ≤ rejected-max（确认够差）→ 写入对
  不用 pairwise：绝对分数比相对排序更适合长度差异大的文本比较

Interviewer（问题质量）：
  生成差问题 → pairwise 直接比较两个问题（问题短，无长度偏见）→ 确认选 A 才写入

输出
────
  data/reports/report_*.json           — 面试报告（SFT 来源）
  data/finetune/dpo_interviewee_*.json — 被面试者偏好对
  data/finetune/dpo_interviewer_*.json — 面试官偏好对
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from finetune.config import finetune_config as ft_cfg
from interview.runner import run_interview
from interview.models import QATurn
from agents.prompts import INTERVIEWEE_SYSTEM, INTERVIEWER_SYSTEM
from utils import extract_json, safe_float


# ─── vLLM 单次调用 ───────────────────────────────────────────────────────────

def _vllm_call(client: OpenAI, system: str, user: str, max_tokens: int = 120) -> str:
    resp = client.chat.completions.create(
        model=ft_cfg.vllm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.9,
    )
    return (resp.choices[0].message.content or "").strip()


# ─── 生成差样本 ───────────────────────────────────────────────────────────────

_BAD_ANSWER_SYS = (
    "你是一个面试表现很差的候选人。"
    "只用1句话作答，内容模糊、没有细节、不提任何具体技术或数字。"
)

_BAD_QUESTION_SYS = (
    "你是一个经验不足的面试官。"
    "提出一个非常宽泛、与职位无关的问题，例如'你觉得自己怎么样？'之类。"
    "只输出问题本身，不超过15字。"
)


def _gen_bad_answer(client: OpenAI, question: str) -> str:
    return _vllm_call(
        client, _BAD_ANSWER_SYS,
        f"面试官：{question}\n\n你的回答（只说一句模糊的话）：",
        max_tokens=60,
    )


def _gen_bad_question(client: OpenAI, candidate: str, position: str) -> str:
    return _vllm_call(
        client, _BAD_QUESTION_SYS,
        f"候选人：{candidate}，应聘：{position}。请提一个宽泛无针对性的问题：",
        max_tokens=40,
    )


# ─── 验证：EvalAgent 给差答案打分 ────────────────────────────────────────────

_SCORE_ONLY_SYS = (
    "你是面试评估员。只返回 JSON：{\"score\": <0-10 的数字>}，不要任何其他内容。"
    "评分标准：内容是否具体、有无实例、技术深度。"
)


def _eval_rejected_score(client: OpenAI, question: str, answer: str) -> float:
    """给差答案打一个简化分数（0-10），不走多维度流程，节省 token。"""
    text = _vllm_call(
        client, _SCORE_ONLY_SYS,
        f"问题：{question[:100]}\n回答：{answer[:150]}",
        max_tokens=20,
    )
    d = extract_json(text) or {}
    return safe_float(d.get("score", 5), lo=0, hi=10)


# ─── 验证：Pairwise 比较两个问题 ─────────────────────────────────────────────

_QUESTION_PAIRWISE_SYS = (
    "你是面试官评审专家。比较两个面试问题的质量，只返回 JSON：\n"
    "{\"preferred\": \"A\" 或 \"B\", \"reason\": \"一句话\"}\n"
    "评判标准：针对性（与候选人背景相关）、深度（能否引导候选人深入作答）、专业性。"
)


def _pairwise_questions(
    client: OpenAI, candidate: str, position: str,
    question_a: str, question_b: str,
) -> str | None:
    """比较两个面试问题，返回 'A' 或 'B'，解析失败返回 None。"""
    text = _vllm_call(
        client, _QUESTION_PAIRWISE_SYS,
        f"候选人背景：{candidate}，应聘：{position}\n"
        f"问题A：{question_a[:120]}\n"
        f"问题B：{question_b[:120]}",
        max_tokens=60,
    )
    d = extract_json(text) or {}
    preferred = d.get("preferred", "")
    return preferred if preferred in ("A", "B") else None


# ─── 偏好对构建 ───────────────────────────────────────────────────────────────

def _build_interviewee_pairs(
    turns: list[QATurn],
    client: OpenAI,
    chosen_min: float,
    rejected_max: float,
) -> tuple[list[dict], dict]:
    """
    Interviewee DPO 对：用绝对分数双边验证。
      chosen:   原始回答 score ≥ chosen_min（EvalAgent 面试中已打）
      rejected: 差回答   score ≤ rejected_max（本函数调用 EvalAgent 验证）
    """
    pairs = []
    stats = {"generated": 0, "confirmed": 0, "rejected_too_good": 0, "parse_err": 0}

    for idx, t in enumerate(turns):
        if t.score < chosen_min or not t.question or not t.answer:
            continue

        stats["generated"] += 1
        print(f"    [EE {idx+1}] 生成差回答...", end=" ", flush=True)

        bad_answer = _gen_bad_answer(client, t.question)
        bad_score  = _eval_rejected_score(client, t.question, bad_answer)

        print(f"差回答得分 {bad_score:.1f}", flush=True)

        if bad_score > rejected_max:
            stats["rejected_too_good"] += 1
            print(f"    [EE {idx+1}] 跳过：差回答得分 {bad_score:.1f} > {rejected_max}")
            continue

        stats["confirmed"] += 1
        pairs.append({
            "system":         INTERVIEWEE_SYSTEM,
            "prompt_user":    f"面试官：{t.question}",
            "chosen":         t.answer,
            "rejected":       bad_answer,
            "chosen_score":   t.score,
            "rejected_score": bad_score,
            "eval_type":      t.eval_type,
        })

    return pairs, stats


def _build_interviewer_pairs(
    turns: list[QATurn],
    candidate: str,
    position: str,
    client: OpenAI,
    chosen_min: float,
) -> tuple[list[dict], dict]:
    """
    Interviewer DPO 对：pairwise 比较两个问题。
      问题短、长度相近，pairwise 有效（无长度偏见）。
    """
    pairs = []
    stats = {"generated": 0, "confirmed": 0, "flipped": 0, "parse_err": 0}
    asked = []

    for idx, t in enumerate(turns):
        question = t.question.strip()
        if not question:
            continue

        # 构建 interviewer prompt（与 data_builder.py 保持一致）
        if idx == 0:
            user_msg = (
                f"候选人：{candidate}，应聘职位：{position}。\n"
                "请生成第一个面试问题，可以是技术题、行为题或情景题。"
            )
        else:
            summary  = "\n".join(f"{i+1}. {asked[i]}" for i in range(idx))
            user_msg = (
                f"候选人：{candidate}，应聘职位：{position}。\n\n"
                f"已提过的问题：\n{summary}\n\n"
                "请继续生成下一个面试问题，注意不要重复，且交替使用不同题型。"
            )

        asked.append(question)

        if t.score < chosen_min:
            continue

        stats["generated"] += 1
        print(f"    [IR {idx+1}] 生成差问题 + pairwise...", end=" ", flush=True)

        bad_q     = _gen_bad_question(client, candidate, position)
        preferred = _pairwise_questions(client, candidate, position, question, bad_q)

        if preferred is None:
            stats["parse_err"] += 1
            print("解析失败", flush=True)
            continue
        if preferred == "B":
            stats["flipped"] += 1
            print(f"翻转跳过（差问题被选为更好）", flush=True)
            continue

        stats["confirmed"] += 1
        print(f"确认 ✓", flush=True)
        pairs.append({
            "system":       INTERVIEWER_SYSTEM,
            "prompt_user":  user_msg,
            "chosen":       question,
            "rejected":     bad_q,
            "chosen_score": t.score,
        })

    return pairs, stats


# ─── 主程序 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成 SFT/DPO 训练数据")
    parser.add_argument("--resume",        required=True)
    parser.add_argument("--jd",            required=True)
    parser.add_argument("--sessions",      type=int,   default=3)
    parser.add_argument("--questions",     type=int,   default=5)
    parser.add_argument("--follow-ups",    type=int,   default=1)
    parser.add_argument("--dpo",           action="store_true")
    parser.add_argument("--chosen-min",    type=float, default=ft_cfg.dpo_chosen_min,
                        help="chosen 最低分（默认 7.0）")
    parser.add_argument("--rejected-max",  type=float, default=ft_cfg.dpo_rejected_max,
                        help="interviewee rejected 最高分（默认 4.0）")
    parser.add_argument("--report-dir",    default=None)
    parser.add_argument("--finetune-dir",  default=None)
    args = parser.parse_args()

    resume_text  = Path(args.resume).read_text(encoding="utf-8")
    jd_text      = Path(args.jd).read_text(encoding="utf-8")
    report_dir   = Path(args.report_dir)   if args.report_dir   else ft_cfg.report_dir
    finetune_dir = Path(args.finetune_dir) if args.finetune_dir else ft_cfg.finetune_data_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.dpo:
        finetune_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=ft_cfg.vllm_base_url, api_key=ft_cfg.vllm_api_key) if args.dpo else None

    all_ee_pairs: list[dict] = []
    all_ir_pairs: list[dict] = []

    for i in range(1, args.sessions + 1):
        print(f"\n{'='*50}")
        print(f"第 {i}/{args.sessions} 场面试")
        print(f"{'='*50}")

        report = run_interview(
            resume_text=resume_text,
            jd_text=jd_text,
            mode="ai",
            num_questions=args.questions,
            num_follow_ups=args.follow_ups,
            verbose=True,
            output_dir=report_dir,
        )

        if args.dpo:
            print(f"\n  生成 DPO 偏好对（共 {len(report.detailed_feedback)} 轮）...")

            ee_pairs, ee_stats = _build_interviewee_pairs(
                report.detailed_feedback, client, args.chosen_min, args.rejected_max,
            )
            ir_pairs, ir_stats = _build_interviewer_pairs(
                report.detailed_feedback, report.candidate_name, report.position,
                client, args.chosen_min,
            )

            all_ee_pairs.extend(ee_pairs)
            all_ir_pairs.extend(ir_pairs)

            print(f"\n  [本场汇总]")
            print(f"  Interviewee: 候选 {ee_stats['generated']}，"
                  f"确认 {ee_stats['confirmed']}，"
                  f"差回答太好跳过 {ee_stats['rejected_too_good']}")
            print(f"  Interviewer: 候选 {ir_stats['generated']}，"
                  f"确认 {ir_stats['confirmed']}，"
                  f"翻转跳过 {ir_stats['flipped']}")

    if args.dpo:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if all_ee_pairs:
            p = finetune_dir / f"dpo_interviewee_{ts}.json"
            p.write_text(json.dumps(all_ee_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nInterviewee DPO → {p}（{len(all_ee_pairs)} 条）")
        else:
            print("\n[警告] Interviewee DPO 无有效偏好对，尝试降低 --rejected-max")

        if all_ir_pairs:
            p = finetune_dir / f"dpo_interviewer_{ts}.json"
            p.write_text(json.dumps(all_ir_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Interviewer DPO → {p}（{len(all_ir_pairs)} 条）")
        else:
            print("[警告] Interviewer DPO 无有效偏好对")

    print("\n数据生成完毕。")


if __name__ == "__main__":
    main()
