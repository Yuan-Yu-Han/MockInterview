"""三个面试 Agent 的工厂函数。

┌──────────────────────────────────────────────────────────────────────┐
│  InterviewerAgent 知道：简历 + JD（+ 可选 RAG 工具）                │
│  IntervieweeAgent 知道：仅自己的简历  ← 刻意信息隔离                │
│  EvalAgent        知道：问题 + 回答；汇总时知道完整面试记录          │
└──────────────────────────────────────────────────────────────────────┘

IntervieweeAgent 信息隔离的设计原因：
  - 真实候选人只了解自己的背景，不知道 JD 内部评分标准或面试官意图
  - 隔离后可以无缝替换为真人参与：
      AI 模式    → 正常调用 create_interviewee()
      人工模式   → runner 跳过此 Agent，直接调用 input()
"""

from typing import Optional, List
from agno.agent import Agent
from agno.models.vllm import VLLM

from config import model
from agents.prompts import (
    INTERVIEWER_SYSTEM,
    INTERVIEWEE_SYSTEM,
    EVALUATOR_SYSTEM,
    PAIRWISE_SYSTEM,
)


def create_model() -> VLLM:
    """创建指向本地 vLLM 服务的 VLLM 模型实例（Qwen3-8B）。"""
    return VLLM(
        id=model.name,
        base_url=model.base_url,
        api_key=model.api_key,
        enable_thinking=False,
        temperature=model.temperature,
        top_p=model.top_p,
    )


def _agent(name: str, system: str, tools: Optional[List] = None) -> Agent:
    """内部辅助函数：用统一配置构造一个 agno Agent。"""
    return Agent(
        name=name,
        model=create_model(),
        tools=tools or [],
        instructions=system,
        markdown=False,
        num_history_runs=0,
        stream=False,
    )


# ─── 三个公开工厂函数 ──────────────────────────────────────────────────────────

def create_interviewer(rag_tools: Optional[List] = None) -> Agent:
    """InterviewerAgent —— 生成面试问题和追问。

    拥有完整上下文（简历 + JD），可使用 RAG 检索具体细节。
    """
    return _agent("InterviewerAgent", INTERVIEWER_SYSTEM, tools=rag_tools)


def create_interviewee(resume_context: str = "") -> Agent:
    """IntervieweeAgent —— 模拟候选人回答问题。

    刻意信息隔离：
      - 系统提示词中只注入自己的简历（resume_context）
      - 不接收 JD、评分标准、面试官意图
      - 与真实面试候选人拥有的信息完全对等

    切换到人工模式时，runner 直接跳过此函数调用 input()，
    其他两个 Agent 完全不需要修改。
    """
    system = INTERVIEWEE_SYSTEM
    if resume_context:
        system = (
            f"{INTERVIEWEE_SYSTEM}\n\n"
            f"--- 你的个人背景（简历） ---\n{resume_context[:600]}"
        )
    return _agent("IntervieweeAgent", system)


def create_evaluator(save_tool=None) -> Agent:
    """EvalAgent —— 对每轮问答打分，面试结束后生成最终报告。

    双模式（由用户消息格式区分）：
      【单题评估】  问题 + 候选人回答 → {eval_type, content_score, structure_score,
                                         relevance_score, score, feedback, ...}
      【最终报告】  完整面试记录     → {overall_score, recommendation, ...}

    若传入 save_tool，EvalAgent 会在生成报告 JSON 后自动调用工具保存文件。

    注意：EvalAgent 永远使用 base 模型，绝对不能参与 SFT/DPO 微调，
    否则会导致评分标准随训练模型漂移，破坏数据飞轮的可靠性。
    """
    tools = [save_tool] if save_tool else []
    return _agent("EvalAgent", EVALUATOR_SYSTEM, tools=tools)


def create_pairwise_evaluator() -> Agent:
    """PairwiseEvalAgent —— 比较两个回答/问题的优劣，输出偏好排序。

    专用于 DPO 数据生成流水线（finetune/generate_data.py）：
      输入：问题 + 回答A + 回答B
      输出：{preferred, confidence, content, structure, relevance, reason}

    排序信号比绝对分数更稳定，适合作为 DPO (chosen, rejected) 对的判据。
    永远使用 base 模型，不参与微调。
    """
    return _agent("PairwiseEvalAgent", PAIRWISE_SYSTEM)
