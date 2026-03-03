"""工具函数：解析模型输出、构建提示词上下文。"""

import json
import re
from typing import Any, Dict, Optional


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型的文本回复中提取第一个合法的 JSON 对象。

    模型可能会在 JSON 前后添加说明文字，或用 markdown 代码块包裹。
    此函数处理以下格式：
      - ```json {...} ```
      - 纯 {...}
      - 说明文字 + {...}
    """
    if not text:
        return None

    # 去掉 markdown 代码块标记
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    # 找到最外层的 {...}
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def safe_list(value: Any, max_items: int = 5) -> list:
    """确保返回值是列表，并截断到 max_items 条。"""
    if isinstance(value, list):
        return value[:max_items]
    if isinstance(value, str):
        return [value]
    return []


def safe_float(value: Any, default: float = 0.0, lo: float = 0.0, hi: float = 100.0) -> float:
    """将 value 强制转换为 float，并限制在 [lo, hi] 范围内。"""
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def build_interview_context(resume_text: str, jd_text: str, max_chars: int = 1200) -> str:
    """构建传给 InterviewerAgent 的紧凑上下文字符串。

    将简历和 JD 各截取一半，合并为一段上下文，
    总长度控制在 max_chars 以内，节省 token 预算。
    """
    resume_snippet = resume_text[: max_chars // 2].strip()
    jd_snippet = jd_text[: max_chars // 2].strip()
    return (
        f"=== 候选人简历 ===\n{resume_snippet}\n\n"
        f"=== 职位描述 ===\n{jd_snippet}"
    )


def build_history_summary(turns: list, last_n: int = 3) -> str:
    """返回最近 N 轮面试的简短摘要，用于提示词中的历史回顾。"""
    if not turns:
        return ""
    recent = turns[-last_n:]
    lines = []
    for i, t in enumerate(recent, 1):
        lines.append(f"Q{i}：{t.question}")
        lines.append(f"A{i}：{t.answer[:120]}...")
    return "\n".join(lines)
