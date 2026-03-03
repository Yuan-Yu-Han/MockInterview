#!/usr/bin/env python3
"""模拟面试系统入口。

用法
────
# 人工练习（你来回答，F5 语音输入或键盘均可）：
python run_interview.py --resume data/resume/sample.txt --jd data/documents/sample_jd.txt --mode human

# AI 全自动模拟：
python run_interview.py --resume ... --jd ...

# 保存报告：
python run_interview.py --resume ... --jd ... --save

环境变量
────────
VLLM_BASE_URL    vLLM 服务地址（默认 http://localhost:8000/v1/）
VLLM_MODEL_NAME  模型名称     （默认 Qwen3-8B）
VLLM_API_KEY     API 密钥     （默认 not-needed）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import interview, paths


def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟面试系统 – agno + vLLM (Qwen3-8B)")
    parser.add_argument("--resume",   required=True, help="简历文件路径（纯文本 / Markdown）")
    parser.add_argument("--jd",       required=True, help="职位描述文件路径")
    parser.add_argument(
        "--mode", choices=["ai", "human"], default="human",
        help="human → 你键盘/语音回答 [默认] | ai → 全自动模拟",
    )
    parser.add_argument(
        "--questions", type=int, default=interview.num_questions,
        help=f"主问题数量（默认 {interview.num_questions}）",
    )
    parser.add_argument(
        "--follow-ups", type=int, default=interview.max_follow_ups,
        choices=[0, 1, 2], dest="follow_ups",
        help=f"每题追问次数 0/1/2（默认 {interview.max_follow_ups}）",
    )
    parser.add_argument("--rag",   action="store_true", default=False, help="启用 RAG 检索")
    parser.add_argument("--save",  action="store_true", default=False, help="保存报告到 data/output/")
    parser.add_argument("--quiet", action="store_true", default=False, help="不打印过程")
    args = parser.parse_args()

    from interview.runner import run_interview
    run_interview(
        resume_text=_read_file(args.resume),
        jd_text=_read_file(args.jd),
        mode=args.mode,
        num_questions=args.questions,
        num_follow_ups=args.follow_ups,
        use_rag=args.rag,
        verbose=not args.quiet,
        output_dir=paths.output if args.save else None,
    )


if __name__ == "__main__":
    main()
