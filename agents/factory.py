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
    INTERVIEWER_RAG_ADDON,
    INTERVIEWEE_SYSTEM,
    EVALUATOR_SYSTEM,
    PAIRWISE_SYSTEM,
)


def _vllm(max_tokens: int = model.max_tokens) -> VLLM:
    """统一构造 VLLM 模型实例，确保 max_tokens 始终生效。

    不传 max_tokens 会让 vLLM 用整个剩余 context 作为输出预算，
    导致 KV cache 耗尽后新请求无限等待（静默挂起）。
    """
    return VLLM(
        id=model.name,
        base_url=model.base_url,
        api_key=model.api_key,
        enable_thinking=False,
        temperature=model.temperature,
        top_p=model.top_p,
        max_tokens=max_tokens,
    )


def create_interviewer(rag_tools: Optional[List] = None) -> Agent:
    """InterviewerAgent —— 生成面试问题和追问。

    Pipeline RAG 模式下，RAG 检索由 _ask() 中的 Python 代码主动执行，
    检索结果已注入 prompt；面试官 agent 只负责改写出题，不持有任何工具。
    rag_tools 参数保留以向后兼容，但不再传给 Agent。
    """
    instructions = INTERVIEWER_SYSTEM + INTERVIEWER_RAG_ADDON
    return Agent(
        name="InterviewerAgent",
        model=_vllm(),
        tools=[],           # Pipeline RAG：agent 无工具，检索由 Python 完成
        instructions=instructions,
        markdown=False,
        num_history_runs=0,
        stream=False,
    )


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
    return Agent(
        name="IntervieweeAgent",
        model=_vllm(),
        tools=[],
        instructions=system,
        markdown=False,
        num_history_runs=0,
        stream=False,
    )


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
    return Agent(
        name="EvalAgent",
        model=_vllm(),
        tools=[save_tool] if save_tool else [],
        instructions=EVALUATOR_SYSTEM,
        markdown=False,
        num_history_runs=0,
        stream=False,
    )


def create_pairwise_evaluator() -> Agent:
    """PairwiseEvalAgent —— 比较两个回答/问题的优劣，输出偏好排序。

    专用于 DPO 数据生成流水线（finetune/generate_data.py）：
      输入：问题 + 回答A + 回答B
      输出：{preferred, confidence, content, structure, relevance, reason}

    排序信号比绝对分数更稳定，适合作为 DPO (chosen, rejected) 对的判据。
    永远使用 base 模型，不参与微调。
    """
    return Agent(
        name="PairwiseEvalAgent",
        model=_vllm(),
        tools=[],
        instructions=PAIRWISE_SYSTEM,
        markdown=False,
        num_history_runs=0,
        stream=False,
    )
