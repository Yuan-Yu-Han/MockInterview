"""模拟面试系统的 Pydantic 数据模型。"""

from typing import List, Literal
from pydantic import BaseModel, Field


class QATurn(BaseModel):
    """面试中的一轮问答记录。"""
    question: str
    question_type: str = "main"      # "main"（主问题）或 "follow_up"（追问）
    answer: str = ""

    # ── 多维度评分（EvalAgent 格式） ──────────────────────────────────────────
    eval_type: str = ""              # "technical" | "behavioral" | "situational"
    content_score: float   = Field(default=0.0, ge=0, le=10)  # 内容质量
    structure_score: float = Field(default=0.0, ge=0, le=10)  # 表达结构
    relevance_score: float = Field(default=0.0, ge=0, le=10)  # 岗位相关性
    score: float           = Field(default=0.0, ge=0, le=10)  # 加权综合分

    feedback: str = ""                                          # 总体评价
    strengths: List[str] = Field(default_factory=list)         # 回答亮点
    areas_for_improvement: List[str] = Field(default_factory=list)  # 待改进点


class PairwiseResult(BaseModel):
    """PairwiseEvalAgent 的比较结果，用于 DPO 数据生成流水线。"""
    preferred:  Literal["A", "B"]
    confidence: Literal["high", "medium", "low"]
    content:    str   # "<A_better|B_better|tie> — <说明>"
    structure:  str
    relevance:  str
    reason:     str   # 1-2 句综合判断

    @property
    def is_reliable(self) -> bool:
        """confidence 为 high 或 medium 时视为可靠偏好对。"""
        return self.confidence in ("high", "medium")


class InterviewSession(BaseModel):
    """一场面试的完整状态，贯穿整个面试流程。"""
    resume_text: str
    jd_text: str
    candidate_name: str = "候选人"
    position: str = "软件工程师"
    turns: List[QATurn] = Field(default_factory=list)  # 所有问答记录
    total_rounds: int = 5                               # 计划主问题数


class InterviewReport(BaseModel):
    """面试结束后的最终评估报告。"""
    candidate_name: str
    position: str
    overall_score: float = Field(ge=0, le=100)          # 综合得分 0-100
    technical_score: float = Field(ge=0, le=10)         # 技术能力 0-10
    communication_score: float = Field(ge=0, le=10)     # 表达沟通 0-10
    questions_asked: int                                 # 实际提问数（含追问）
    key_strengths: List[str] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)
    recommendation: str = "Maybe"  # Strong Hire | Hire | Maybe | No Hire
    improvement_suggestions: List[str] = Field(default_factory=list)
    detailed_feedback: List[QATurn] = Field(default_factory=list)  # 逐题反馈
