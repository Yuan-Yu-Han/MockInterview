"""面试主流程编排。

两种模式
────────
  "ai"    → InterviewerAgent ↔ IntervieweeAgent（完整 AI 模拟）
  "human" → InterviewerAgent ↔ 你本人          （人工练习模式）

人工模式下 IntervieweeAgent 完全不参与——runner 在每个问题后等待键盘输入。
InterviewerAgent 和 EvalAgent 在两种模式下行为完全一致。

流程
────
1. 面试循环（N 轮主问题，每轮可 0-2 次追问）：
     InterviewerAgent  →  问题
     IntervieweeAgent 或 input()  →  回答
     EvalAgent         →  打分 + 反馈（面试过程中对候选人不可见）
2. EvalAgent × 1 次调用   → InterviewReport（+ 可选自动保存）
"""

import asyncio
from pathlib import Path
from typing import Literal, Optional

from agno.agent import Agent
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from config import interview, mcp
from .models import InterviewReport, InterviewSession, QATurn
from agents import create_interviewer, create_interviewee, create_evaluator
from agents.prompts import REPORT_MSG
from utils import build_history_summary, build_interview_context, extract_json, safe_float, safe_list
from tools.report import make_save_tool

InterviewMode = Literal["ai", "human"]

console = Console()


# ─── 面试循环 ──────────────────────────────────────────────────────────────────

# ─── Pipeline RAG：Python 主动检索，不依赖 LLM 决策 ──────────────────────────

# 面试话题轮转：按题号循环，确保每场面试覆盖不同维度
_TOPIC_ROTATION = [
    "RAG系统设计 向量数据库",
    "LLM应用架构 Agent工具调用",
    "模型部署 vLLM 推理优化",
    "提示工程 结构化输出",
    "项目经验 系统设计",
    "行为题 问题解决 学习能力",
]


def _extract_rag_query(context: str, question_num: int) -> str:
    """从上下文提取 RAG 检索关键词。

    策略：
    1. 分离简历和 JD 两段文本（build_interview_context 的固定格式）
    2. 分别提取技术关键词，优先查"候选人有 ∩ JD 要求"的交集
       → 交集命中率高、问题针对性强
       → 无交集则退回候选人已有技能（能答上来）
    3. 叠加题号轮转话题，确保每场面试覆盖不同维度
    """
    tech_kw = [
        "RAG", "LLM", "Agent", "MCP", "vLLM", "ChromaDB", "向量数据库",
        "FastAPI", "Python", "系统设计", "微服务", "提示工程", "模型部署",
        "Embedding", "HNSW", "BM25", "Reranker", "Transformer", "Fine-tuning",
        "微调", "量化", "RLHF", "function calling", "工具调用",
    ]

    # 分离简历与 JD（build_interview_context 的固定分隔格式）
    resume_part = context
    jd_part = ""
    if "=== 职位描述 ===" in context:
        sections = context.split("=== 职位描述 ===")
        resume_part = sections[0].replace("=== 候选人简历 ===", "").strip()
        jd_part = sections[1].strip() if len(sections) > 1 else ""

    resume_skills = [kw for kw in tech_kw if kw.lower() in resume_part.lower()]
    jd_needs      = [kw for kw in tech_kw if kw.lower() in jd_part.lower()]

    # 优先：候选人有 ∩ JD 要求（双方都关心）
    intersection = [kw for kw in resume_skills if kw in jd_needs]
    priority = intersection or resume_skills or jd_needs or ["AI应用开发"]

    tech_part  = " ".join(priority[:2])
    topic_part = _TOPIC_ROTATION[(question_num - 1) % len(_TOPIC_ROTATION)]
    return f"{tech_part} {topic_part}"


def _rag_retrieve(mcp_instance, query: str, loop, top_k: int = 2) -> str:
    """直接通过 mcp_instance.session.call_tool 调用 query_knowledge_hub。

    绕过 LLM 工具调用决策，由 Python 强制执行检索。
    返回检索到的题目文本，供注入 prompt 使用。
    """
    async def _coro():
        result = await mcp_instance.session.call_tool(
            "query_knowledge_hub",
            arguments={
                "query": query,
                "collection": "knowledge_hub",
                "n_results": top_k,
            },
        )
        texts = []
        for item in (result.content or []):
            t = getattr(item, "text", None)
            if t:
                texts.append(t)
        return "\n\n---\n\n".join(texts)

    fut = asyncio.run_coroutine_threadsafe(_coro(), loop)
    try:
        return fut.result(timeout=15)
    except Exception as e:
        print(f"[RAG] 检索失败: {e}", flush=True)
        return ""


def _ask(
    interviewer: Agent,
    context: str,
    history_summary: str,
    question_num: int,
    is_follow_up: bool = False,
    prev_answer: str = "",
    prev_score: float = 0.0,
    prev_feedback: str = "",
    _mcp=None,   # MCPTools instance；提供时 Python 主动检索（Pipeline RAG）
    _loop=None,  # asyncio event loop；与 _mcp 配套使用
) -> tuple:
    """调用 InterviewerAgent 生成主问题或追问。

    Pipeline RAG 模式（_mcp + _loop 均不为 None）：
        1. Python 提取关键词 → 调 MCP query_knowledge_hub → 得到参考题目
        2. 参考题目注入 prompt
        3. 面试官 agent 仅做改写，无需持有任何工具 → 始终用 sync run()

    Returns:
        (question_text: str, rag_hit: bool)
        rag_hit=True 表示本次成功检索到题库内容。
    """
    # ── Step 1：Pipeline RAG 检索（主问题时触发，追问时跳过）────────────────
    rag_context = ""
    rag_hit = False
    if _mcp is not None and _loop is not None and not is_follow_up:
        query = _extract_rag_query(context, question_num)
        print(f"[RAG] 检索关键词: {query!r}", flush=True)
        rag_context = _rag_retrieve(_mcp, query, _loop)
        rag_hit = bool(rag_context)
        if rag_hit:
            print(f"[RAG] 检索成功，注入 {len(rag_context)} 字参考内容", flush=True)
        else:
            print("[RAG] 检索未返回结果，面试官将自主出题", flush=True)

    # ── Step 2：构建 prompt ───────────────────────────────────────────────────
    if is_follow_up:
        prompt = (
            f"{context}\n\n"
            f"候选人刚才的回答：\n\"{prev_answer}\"\n\n"
            f"评分：{prev_score:.1f}/10\n"
            f"评估反馈：{prev_feedback}\n\n"
            "请根据以上评分和反馈，针对候选人回答较弱或值得深挖的点生成一个追问。"
        )
    else:
        history_block = f"已问过的问题（避免重复）：\n{history_summary}\n\n" if history_summary else ""
        rag_block = (
            f"【题库参考】\n{rag_context}\n\n"
            if rag_context else ""
        )
        guide = (
            "出题步骤：\n"
            "1. 从候选人简历中找一个具体项目或技术经历作为问题锚点\n"
            "2. 参考题库的考察要点，围绕这个锚点设计考察问题\n"
            "3. 确保问题与职位描述的核心要求相关\n"
            "只输出问题本身（1-3句话），不要输出分析过程。"
            if rag_context else
            f"请生成第 {question_num} 个面试问题，结合候选人简历和职位要求，"
            "只输出问题本身，不要任何其他内容。"
        )
        prompt = (
            f"{context}\n\n"
            f"{history_block}"
            f"{rag_block}"
            f"{guide}"
        )

    # ── Step 3：调用面试官 agent（始终 sync run，无工具）────────────────────
    resp = interviewer.run(prompt)
    text = (resp.content or "").strip()
    if "cancelled" in text.lower():
        raise KeyboardInterrupt
    return text, rag_hit


def _answer_ai(interviewee: Agent, question: str) -> str:
    """IntervieweeAgent 回答问题（简历已注入系统提示词，无需再次传入）。"""
    resp = interviewee.run(f"面试官：{question}\n\n你的回答：")
    text = (resp.content or "").strip()
    if "cancelled" in text.lower():
        raise KeyboardInterrupt
    return text


def _answer_human(question: str, label: str) -> str:
    """人工模式：展示问题，单次 Enter 提交回答。"""
    console.print()
    console.print(Panel(
        Text(question, style="white"),
        title=f"[bold cyan]面试官  ·  第 {label} 题[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print("[dim]输入完成后按 Enter 提交[/dim]\n")

    try:
        answer = input("  ❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt

    if not answer:
        answer = "（未作答）"

    # 回显：让用户确认识别的文字
    console.print()
    console.print(Panel(
        Text(answer, style="white"),
        title="[dim]你的回答[/dim]",
        border_style="dim",
        padding=(0, 2),
    ))
    return answer


def _evaluate(evaluator: Agent, question: str, answer: str) -> QATurn:
    """调用 EvalAgent 对一轮问答打分，返回填充好的 QATurn。"""
    prompt = f"问题：{question}\n\n候选人回答：{answer}\n\n请评估："
    resp = evaluator.run(prompt)
    d = extract_json(resp.content if resp else "") or {}
    return QATurn(
        question=question,
        answer=answer,
        eval_type=d.get("eval_type", ""),
        content_score=safe_float(d.get("content_score", 5), lo=0, hi=10),
        structure_score=safe_float(d.get("structure_score", 5), lo=0, hi=10),
        relevance_score=safe_float(d.get("relevance_score", 5), lo=0, hi=10),
        score=safe_float(d.get("score", 5), lo=0, hi=10),
        feedback=d.get("feedback", ""),
        strengths=safe_list(d.get("strengths", [])),
        areas_for_improvement=safe_list(d.get("areas_for_improvement", [])),
    )


# ─── 报告生成 ──────────────────────────────────────────────────────────────────

def _generate_report(
    evaluator: Agent,
    session: InterviewSession,
    output_dir: Optional[Path] = None,
) -> InterviewReport:
    transcript = "\n".join(
        f"Q：{t.question}\nA：{t.answer[:200]}\n得分：{t.score}/10"
        for t in session.turns
    )
    msg = REPORT_MSG.format(
        name=session.candidate_name,
        title=session.position,
        transcript=transcript,
    )
    if output_dir:
        msg += "\n\n请在生成报告 JSON 后立即调用 save_report 工具将其保存到文件。"

    resp = evaluator.run(msg)
    d = extract_json(resp.content if resp else "") or {}

    avg = sum(t.score for t in session.turns) / len(session.turns) if session.turns else 0.0
    return InterviewReport(
        candidate_name=session.candidate_name,
        position=session.position,
        overall_score=safe_float(d.get("overall_score", avg * 10), lo=0, hi=100),
        technical_score=safe_float(d.get("technical_score", avg), lo=0, hi=10),
        communication_score=safe_float(d.get("communication_score", avg), lo=0, hi=10),
        questions_asked=len(session.turns),
        key_strengths=safe_list(d.get("key_strengths", [])),
        skill_gaps=safe_list(d.get("skill_gaps", [])),
        recommendation=d.get("recommendation", "Maybe"),
        improvement_suggestions=safe_list(d.get("improvement_suggestions", [])),
        detailed_feedback=session.turns,
    )


# ─── 公开接口 ──────────────────────────────────────────────────────────────────

def run_interview(
    resume_text: str,
    jd_text: str,
    mode: InterviewMode = "ai",
    num_questions: int = interview.num_questions,
    num_follow_ups: int = interview.max_follow_ups,
    use_rag: bool = False,
    verbose: bool = True,
    output_dir: Optional[Path] = None,
) -> InterviewReport:
    if use_rag:
        return asyncio.run(
            _run_with_rag(resume_text, jd_text, mode, num_questions, num_follow_ups, verbose, output_dir)
        )
    return _run(resume_text, jd_text, mode, num_questions, num_follow_ups, verbose, output_dir)


# ─── 核心循环（无 RAG）────────────────────────────────────────────────────────

def _run(
    resume_text: str,
    jd_text: str,
    mode: InterviewMode,
    num_questions: int,
    num_follow_ups: int,
    verbose: bool,
    output_dir: Optional[Path],
) -> InterviewReport:

    candidate_name = resume_text.strip().splitlines()[0][:50] if resume_text.strip() else "候选人"
    position = jd_text.strip().splitlines()[0][:50] if jd_text.strip() else "软件工程师"

    if verbose:
        _header(candidate_name, position, mode, num_questions, num_follow_ups)

    session = InterviewSession(
        resume_text=resume_text,
        jd_text=jd_text,
        candidate_name=candidate_name,
        position=position,
        total_rounds=num_questions,
    )

    save_tool = make_save_tool(output_dir, session) if output_dir else None
    evaluator = create_evaluator(save_tool=save_tool)

    if verbose and mode == "human":
        console.print("[dim]提示：F5 语音输入或直接键盘输入，按 Enter 提交[/dim]\n")

    interviewer = create_interviewer()
    interviewee = create_interviewee(resume_context=resume_text) if mode == "ai" else None
    context = build_interview_context(resume_text, jd_text)

    try:
        for i in range(1, num_questions + 1):
            history = build_history_summary(session.turns, last_n=2)

            if verbose and mode == "ai":
                console.print(f"\n[dim]生成第 {i}/{num_questions} 题...[/dim]")

            question, _ = _ask(interviewer, context, history, question_num=i)

            if mode == "ai":
                answer = _answer_ai(interviewee, question)
            else:
                answer = _answer_human(question, label=f"{i}/{num_questions}")

            if verbose and mode == "ai":
                console.print(f"\n[dim]评估回答...[/dim]")

            turn = _evaluate(evaluator, question, answer)
            turn.question_type = "main"
            session.turns.append(turn)

            if verbose:
                if mode == "ai":
                    _print_turn_ai(i, question, answer, turn.score, turn.feedback)
                else:
                    _print_eval(turn.score, turn.feedback)

            # 追问
            cur_answer, cur_turn = answer, turn
            for j in range(1, num_follow_ups + 1):
                label = f"{i}↳" if num_follow_ups == 1 else f"{i}↳{j}"

                fu_q, _ = _ask(
                    interviewer, context, "", i,
                    is_follow_up=True,
                    prev_answer=cur_answer,
                    prev_score=cur_turn.score,
                    prev_feedback=cur_turn.feedback,
                )

                if mode == "ai":
                    fu_a = _answer_ai(interviewee, fu_q)
                else:
                    fu_a = _answer_human(fu_q, label=label)

                fu_turn = _evaluate(evaluator, fu_q, fu_a)
                fu_turn.question_type = "follow_up"
                session.turns.append(fu_turn)

                if verbose:
                    if mode == "ai":
                        _print_turn_ai(label, fu_q, fu_a, fu_turn.score, fu_turn.feedback)
                    else:
                        _print_eval(fu_turn.score, fu_turn.feedback)

                cur_answer, cur_turn = fu_a, fu_turn

    except KeyboardInterrupt:
        if verbose:
            console.print("\n[yellow]已中断，根据已完成题目生成报告...[/yellow]")
        if not session.turns:
            raise

    if verbose:
        console.print(Rule("[dim]生成评估报告[/dim]"))

    report = _generate_report(evaluator, session, output_dir)
    if verbose:
        _print_report(report)
    return report


# ─── 带 RAG 的循环 ────────────────────────────────────────────────────────────

async def _run_with_rag(
    resume_text: str,
    jd_text: str,
    mode: InterviewMode,
    num_questions: int,
    num_follow_ups: int,
    verbose: bool,
    output_dir: Optional[Path],
) -> InterviewReport:
    from agno.tools.mcp import MCPTools

    candidate_name = resume_text.strip().splitlines()[0][:50] if resume_text.strip() else "候选人"
    position = jd_text.strip().splitlines()[0][:50] if jd_text.strip() else "软件工程师"

    if verbose:
        _header(candidate_name, position, mode, num_questions, num_follow_ups, rag=True)
        console.print("[dim]正在连接 RAG MCP 服务器...[/dim]")

    async with MCPTools(
        command=mcp.command,
        transport="stdio",
        include_tools=["query_knowledge_hub", "list_collections"],
    ) as rag_tools:

        session = InterviewSession(
            resume_text=resume_text, jd_text=jd_text,
            candidate_name=candidate_name, position=position, total_rounds=num_questions,
        )

        save_tool = make_save_tool(output_dir, session) if output_dir else None
        evaluator = create_evaluator(save_tool=save_tool)

        interviewer = create_interviewer(rag_tools=[rag_tools])
        interviewee = create_interviewee(resume_context=resume_text) if mode == "ai" else None
        context = build_interview_context(resume_text, jd_text)

        try:
            for i in range(1, num_questions + 1):
                history = build_history_summary(session.turns, last_n=2)
                question, _ = _ask(interviewer, context, history, i)
                answer = _answer_ai(interviewee, question) if mode == "ai" else _answer_human(question, f"{i}/{num_questions}")
                turn = _evaluate(evaluator, question, answer)
                turn.question_type = "main"
                session.turns.append(turn)

                if verbose:
                    if mode == "ai":
                        _print_turn_ai(i, question, answer, turn.score, turn.feedback)
                    else:
                        _print_eval(turn.score, turn.feedback)

                cur_answer, cur_turn = answer, turn
                for j in range(1, num_follow_ups + 1):
                    label = f"{i}↳" if num_follow_ups == 1 else f"{i}↳{j}"
                    fu_q, _ = _ask(interviewer, context, "", i, is_follow_up=True,
                                   prev_answer=cur_answer, prev_score=cur_turn.score, prev_feedback=cur_turn.feedback)
                    fu_a = _answer_ai(interviewee, fu_q) if mode == "ai" else _answer_human(fu_q, label)
                    fu_turn = _evaluate(evaluator, fu_q, fu_a)
                    fu_turn.question_type = "follow_up"
                    session.turns.append(fu_turn)

                    if verbose:
                        if mode == "ai":
                            _print_turn_ai(label, fu_q, fu_a, fu_turn.score, fu_turn.feedback)
                        else:
                            _print_eval(fu_turn.score, fu_turn.feedback)

                    cur_answer, cur_turn = fu_a, fu_turn

        except KeyboardInterrupt:
            if verbose:
                console.print("\n[yellow]已中断，生成报告...[/yellow]")
            if not session.turns:
                raise

        if verbose:
            console.print(Rule("[dim]生成评估报告[/dim]"))
        report = _generate_report(evaluator, session, output_dir)
        if verbose:
            _print_report(report)
        return report


# ─── 终端展示辅助函数 ─────────────────────────────────────────────────────────

def _header(
    candidate_name: str,
    position: str,
    mode: InterviewMode,
    num_questions: int,
    num_follow_ups: int,
    rag: bool = False,
) -> None:
    mode_str = "AI 模拟" if mode == "ai" else "人工练习"
    rag_str  = " · RAG" if rag else ""
    fu_str   = f" · 追问 ×{num_follow_ups}" if num_follow_ups else ""
    console.print()
    console.print(Rule(f"[bold]模拟面试系统[/bold]  [dim]{mode_str}{rag_str}[/dim]"))
    console.print(f"  [cyan]{candidate_name}[/cyan]  ·  [cyan]{position}[/cyan]"
                  f"  [dim]| {num_questions} 题{fu_str}[/dim]")
    console.print(Rule())


def _score_color(score: float) -> str:
    if score >= 8:
        return "green"
    if score >= 6:
        return "yellow"
    return "red"


def _print_eval(score: float, feedback: str) -> None:
    color = _score_color(score)
    console.print()
    console.print(Panel(
        f"[{color}]● {score:.1f} / 10[/{color}]   {feedback}",
        title="[dim]评分反馈[/dim]",
        border_style="dim",
        padding=(0, 2),
    ))


def _print_turn_ai(label, question: str, answer: str, score: float, feedback: str) -> None:
    color = _score_color(score)
    console.print(Rule(f"[dim]第 {label} 题[/dim]"))
    console.print(f"[cyan]面试官[/cyan]  {question}")
    console.print(f"[green]候选人[/green]  {answer}")
    console.print(f"[{color}]● {score:.1f}/10[/{color}]  [dim]{feedback}[/dim]")


def _print_report(report: InterviewReport) -> None:
    rec_color = {
        "Strong Hire": "green",
        "Hire":        "cyan",
        "Maybe":       "yellow",
        "No Hire":     "red",
    }.get(report.recommendation, "white")

    console.print()
    console.print(Rule("[bold]面试评估报告[/bold]"))

    # 分数概览
    t = Table.grid(padding=(0, 3))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("候选人",  f"[cyan]{report.candidate_name}[/cyan]")
    t.add_row("应聘职位", f"[cyan]{report.position}[/cyan]")
    t.add_row("录用建议", f"[{rec_color}]{report.recommendation}[/{rec_color}]")
    t.add_row("综合得分", f"[bold]{report.overall_score:.0f}[/bold] / 100")
    t.add_row("技术能力", f"{report.technical_score:.1f} / 10")
    t.add_row("沟通表达", f"{report.communication_score:.1f} / 10")
    console.print(t)

    if report.key_strengths:
        console.print(f"\n[green]核心优势[/green]")
        for s in report.key_strengths:
            console.print(f"  [green]✓[/green] {s}")

    if report.skill_gaps:
        console.print(f"\n[red]技能缺口[/red]")
        for s in report.skill_gaps:
            console.print(f"  [red]✗[/red] {s}")

    if report.improvement_suggestions:
        console.print(f"\n[yellow]改进建议[/yellow]")
        for s in report.improvement_suggestions:
            console.print(f"  [yellow]→[/yellow] {s}")

    console.print()
    console.print(Rule())
