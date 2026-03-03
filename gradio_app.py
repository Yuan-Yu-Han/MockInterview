"""Gradio 前端：Qwen3-ASR 语音识别 + MockInterview 面试流程。

启动：
    python gradio_app.py
访问：http://localhost:7862
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import gradio as gr
from qwen_asr import Qwen3ASRModel

from interview.runner import _ask, _evaluate, _generate_report
from interview.models import InterviewSession
from agents import create_interviewer, create_evaluator
from utils import build_interview_context, build_history_summary
from config import interview as interview_cfg

# ─── ASR 模型（启动时加载一次）────────────────────────────────────────────────

MODEL_PATH = "/projects/yuan0165/Qwen3-ASR-0.6B"

print(f"Loading Qwen3-ASR from {MODEL_PATH} ...")
asr_model = Qwen3ASRModel.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=4,
    max_new_tokens=512,
)
print("ASR model ready.\n")


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────

def _transcribe(audio_path: str) -> str:
    if not audio_path:
        return ""
    results = asr_model.transcribe(audio=audio_path, language=None)
    return (results[0].text or "").strip()


def _is_last_question(state: dict) -> bool:
    """判断当前题是否为最后一题（含追问）。"""
    at_last_main = state["current_q_idx"] >= state["num_questions"]
    no_more_followup = (
        state["is_follow_up"]
        or state["num_follow_ups"] == 0
        or state["follow_up_count"] >= state["num_follow_ups"]
    )
    return at_last_main and no_more_followup


def _format_eval(turn) -> str:
    score = turn.score
    emoji = "🟢" if score >= 8 else ("🟡" if score >= 6 else "🔴")
    lines = [f"{emoji} **{score:.1f} / 10**", "", turn.feedback or ""]
    if turn.strengths:
        lines += ["", "**优点：** " + "、".join(turn.strengths)]
    if turn.areas_for_improvement:
        lines += ["", "**改进：** " + "、".join(turn.areas_for_improvement)]
    return "\n".join(lines)


def _format_report(report) -> str:
    rec_emoji = {"Strong Hire": "🟢", "Hire": "🔵", "Maybe": "🟡", "No Hire": "🔴"}.get(
        report.recommendation, "⚪"
    )
    lines = [
        "## 面试评估报告",
        "",
        "| 项目 | 结果 |",
        "|------|------|",
        f"| 候选人 | {report.candidate_name} |",
        f"| 应聘职位 | {report.position} |",
        f"| 录用建议 | {rec_emoji} **{report.recommendation}** |",
        f"| 综合得分 | **{report.overall_score:.0f} / 100** |",
        f"| 技术能力 | {report.technical_score:.1f} / 10 |",
        f"| 沟通表达 | {report.communication_score:.1f} / 10 |",
    ]
    if report.key_strengths:
        lines += ["", "### 核心优势"] + [f"- {s}" for s in report.key_strengths]
    if report.skill_gaps:
        lines += ["", "### 技能缺口"] + [f"- {s}" for s in report.skill_gaps]
    if report.improvement_suggestions:
        lines += ["", "### 改进建议"] + [f"- {s}" for s in report.improvement_suggestions]
    return "\n".join(lines)


# ─── 初始/重置 state ───────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "phase": "setup",
        "session": None,
        "interviewer": None,
        "evaluator": None,
        "context": "",
        "current_q_idx": 0,
        "is_follow_up": False,
        "follow_up_count": 0,
        "num_questions": interview_cfg.num_questions,
        "num_follow_ups": interview_cfg.max_follow_ups,
        "current_question": "",
        "last_answer": "",
        "last_score": 0.0,
        "last_feedback": "",
    }


# ─── 固定 11 个 outputs 的辅助构造器 ──────────────────────────────────────────
# 顺序: state, setup_col, interview_col, progress_md, question_md,
#       answer_box, eval_col(visible), eval_md(value), next_btn(value),
#       report_col(visible), report_md(value)
# eval_col 和 report_col 均为顶层 Column，避免嵌套 visible 切换不可靠的问题

def _outputs(
    state,
    setup_visible=None,
    interview_visible=None,
    progress=None,
    question=None,
    answer_value=None,
    answer_interactive=None,
    eval_col_visible=None,    # 控制 eval_col 整体显隐
    eval_value=None,          # eval_md 内容
    next_value=None,          # next_btn 标签
    report_col_visible=None,  # 控制 report_col 整体显隐
    report_value=None,        # report_md 内容
):
    setup_upd      = gr.update() if setup_visible is None      else gr.update(visible=setup_visible)
    interview_upd  = gr.update() if interview_visible is None  else gr.update(visible=interview_visible)
    progress_upd   = gr.update() if progress is None           else gr.update(value=progress)
    question_upd   = gr.update() if question is None           else gr.update(value=question)
    eval_col_upd   = gr.update() if eval_col_visible is None   else gr.update(visible=eval_col_visible)
    eval_upd       = gr.update() if eval_value is None         else gr.update(value=eval_value)
    next_upd       = gr.update() if next_value is None         else gr.update(value=next_value)
    report_col_upd = gr.update() if report_col_visible is None else gr.update(visible=report_col_visible)
    report_upd     = gr.update() if report_value is None       else gr.update(value=report_value)

    if answer_value is None and answer_interactive is None:
        answer_upd = gr.update()
    elif answer_value is None:
        answer_upd = gr.update(interactive=answer_interactive)
    elif answer_interactive is None:
        answer_upd = gr.update(value=answer_value)
    else:
        answer_upd = gr.update(value=answer_value, interactive=answer_interactive)

    return (
        state,
        setup_upd, interview_upd,
        progress_upd, question_upd,
        answer_upd, eval_col_upd, eval_upd, next_upd,
        report_col_upd, report_upd,
    )


# ─── 事件处理函数 ──────────────────────────────────────────────────────────────

def start_interview(resume: str, jd: str, num_q: int, num_fu: int, state: dict):
    if not resume.strip() or not jd.strip():
        gr.Warning("请填写简历和职位描述后再开始。")
        return _outputs(state)

    candidate_name = resume.strip().splitlines()[0][:50]
    position = jd.strip().splitlines()[0][:50]

    session = InterviewSession(
        resume_text=resume,
        jd_text=jd,
        candidate_name=candidate_name,
        position=position,
        total_rounds=num_q,
    )
    interviewer = create_interviewer()
    evaluator   = create_evaluator()
    context     = build_interview_context(resume, jd)

    state = _empty_state()
    state.update({
        "phase": "questioning",
        "session": session,
        "interviewer": interviewer,
        "evaluator": evaluator,
        "context": context,
        "current_q_idx": 1,
        "num_questions": num_q,
        "num_follow_ups": num_fu,
    })

    question = _ask(interviewer, context, "", question_num=1)
    state["current_question"] = question

    label = f"第 1 / {num_q} 题"
    return _outputs(
        state,
        setup_visible=False,
        interview_visible=True,
        progress=f"**{label}**",
        question=question,
        answer_value="",
        answer_interactive=True,
        eval_col_visible=False,
        report_col_visible=False,
    )


def on_audio_stop(audio_path: str):
    """麦克风停止录音后自动 ASR，填入回答文本框。"""
    if not audio_path:
        return gr.update()
    text = _transcribe(audio_path)
    return gr.update(value=text)


def submit_answer(answer: str, state: dict):
    print(f"[submit_answer] phase={state.get('phase') if state else None!r}")

    if state is None:
        return _outputs(_empty_state(),
                        eval_col_visible=True,
                        eval_value="⚠️ 状态丢失，请刷新页面重新开始。")

    if state.get("phase") != "questioning":
        return _outputs(state,
                        eval_col_visible=True,
                        eval_value=f"⚠️ 当前状态 phase={state.get('phase')!r}，无法提交（请先点击开始面试）。")

    answer = answer.strip() or "（未作答）"

    try:
        question  = state["current_question"]
        evaluator = state["evaluator"]

        turn = _evaluate(evaluator, question, answer)
        turn.question_type = "follow_up" if state["is_follow_up"] else "main"
        state["session"].turns.append(turn)

        state["last_answer"]   = answer
        state["last_score"]    = turn.score
        state["last_feedback"] = turn.feedback
        state["phase"]         = "evaluating"

        is_last    = _is_last_question(state)
        next_label = "生成报告" if is_last else "下一题"

        return _outputs(
            state,
            answer_interactive=False,
            eval_col_visible=True,
            eval_value=_format_eval(turn),
            next_value=next_label,
        )
    except Exception as exc:
        print(f"[submit_answer] ERROR: {exc}")
        return _outputs(
            state,
            eval_col_visible=True,
            eval_value=f"❌ 评估出错：{exc}",
        )


def next_question(state: dict):
    if state is None or state.get("phase") != "evaluating":
        return _outputs(state)

    session    = state["session"]
    interviewer = state["interviewer"]
    evaluator   = state["evaluator"]

    # 分支 1：生成追问
    can_follow_up = (
        not state["is_follow_up"]
        and state["num_follow_ups"] > 0
        and state["follow_up_count"] < state["num_follow_ups"]
    )
    if can_follow_up:
        fu_q = _ask(
            interviewer, state["context"], "",
            state["current_q_idx"],
            is_follow_up=True,
            prev_answer=state["last_answer"],
            prev_score=state["last_score"],
            prev_feedback=state["last_feedback"],
        )
        state["current_question"] = fu_q
        state["is_follow_up"]     = True
        state["follow_up_count"] += 1
        state["phase"] = "questioning"

        q_idx = state["current_q_idx"]
        num_q = state["num_questions"]
        label = f"第 {q_idx} / {num_q} 题（追问）"
        return _outputs(
            state,
            progress=f"**{label}**",
            question=fu_q,
            answer_value="",
            answer_interactive=True,
            eval_col_visible=False,
        )

    # 分支 2：下一道主题
    if state["current_q_idx"] < state["num_questions"]:
        state["current_q_idx"]  += 1
        state["is_follow_up"]    = False
        state["follow_up_count"] = 0
        state["phase"] = "questioning"

        history  = build_history_summary(session.turns, last_n=2)
        question = _ask(
            interviewer, state["context"], history,
            state["current_q_idx"],
        )
        state["current_question"] = question

        q_idx = state["current_q_idx"]
        num_q = state["num_questions"]
        label = f"第 {q_idx} / {num_q} 题"
        return _outputs(
            state,
            progress=f"**{label}**",
            question=question,
            answer_value="",
            answer_interactive=True,
            eval_col_visible=False,
        )

    # 分支 3：全部完成，生成报告
    state["phase"] = "report"
    try:
        report = _generate_report(evaluator, session)
        return _outputs(
            state,
            interview_visible=False,
            eval_col_visible=False,
            report_col_visible=True,
            report_value=_format_report(report),
        )
    except Exception as exc:
        print(f"[next_question/report] ERROR: {exc}")
        state["phase"] = "evaluating"  # 回退，让用户可以重试
        return _outputs(
            state,
            eval_col_visible=True,
            eval_value=f"❌ 报告生成出错：{exc}",
            next_value="生成报告",
        )


def restart(state: dict):
    return _outputs(
        _empty_state(),
        setup_visible=True,
        interview_visible=False,
        progress="",
        question="",
        answer_value="",
        answer_interactive=True,
        eval_col_visible=False,
        eval_value="",
        next_value="下一题",
        report_col_visible=False,
        report_value="",
    )


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="MockInterview", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ MockInterview — AI 模拟面试")

    state = gr.State(_empty_state())

    # ── Setup 区 ──────────────────────────────────────────────────────────────
    with gr.Column(visible=True) as setup_col:
        gr.Markdown("### 配置面试")
        with gr.Row():
            resume_box = gr.Textbox(
                label="简历内容（粘贴全文，第一行为候选人姓名）",
                lines=8,
                placeholder="张三\n5年Python后端开发经验\n...",
            )
            jd_box = gr.Textbox(
                label="职位描述（第一行为职位名称）",
                lines=8,
                placeholder="高级后端工程师\n负责核心服务开发...\n要求：Python, Go, 分布式系统",
            )
        with gr.Row():
            num_q_slider = gr.Slider(
                minimum=3, maximum=8, step=1,
                value=interview_cfg.num_questions,
                label="面试题数",
            )
            num_fu_dropdown = gr.Dropdown(
                choices=[0, 1, 2],
                value=interview_cfg.max_follow_ups,
                label="每题追问次数",
            )
        start_btn = gr.Button("开始面试", variant="primary", size="lg")

    # ── Interview 区 ──────────────────────────────────────────────────────────
    with gr.Column(visible=False) as interview_col:
        progress_md = gr.Markdown("**第 1 / 5 题**")
        question_md = gr.Markdown("")

        gr.Markdown("---")
        gr.Markdown("**录音回答**（停止后自动识别）或直接在下方输入文字：")
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎤 点击开始录音",
        )
        answer_box = gr.Textbox(
            label="回答内容（可在此编辑识别结果）",
            lines=4,
            placeholder="语音识别结果将自动填入，也可直接输入…",
            interactive=True,
        )
        submit_btn = gr.Button("提交回答", variant="primary")

    # ── Eval 区（顶层，不嵌套在 interview_col 内，visibility 切换更可靠）────
    with gr.Column(visible=False) as eval_col:
        gr.Markdown("---")
        eval_md  = gr.Markdown("")
        next_btn = gr.Button("下一题", variant="secondary")

    # ── Report 区（顶层）─────────────────────────────────────────────────────
    with gr.Column(visible=False) as report_col:
        report_md   = gr.Markdown("")
        restart_btn = gr.Button("重新开始", variant="secondary")

    # ── 所有事件的公共 outputs 列表（11 项）──────────────────────────────────
    OUTPUTS = [
        state,
        setup_col, interview_col,
        progress_md, question_md,
        answer_box, eval_col, eval_md, next_btn,
        report_col, report_md,
    ]

    # ── 事件绑定 ──────────────────────────────────────────────────────────────
    start_btn.click(
        fn=start_interview,
        inputs=[resume_box, jd_box, num_q_slider, num_fu_dropdown, state],
        outputs=OUTPUTS,
    )

    audio_input.stop_recording(
        fn=on_audio_stop,
        inputs=[audio_input],
        outputs=[answer_box],
    )

    submit_btn.click(
        fn=submit_answer,
        inputs=[answer_box, state],
        outputs=OUTPUTS,
    )

    next_btn.click(
        fn=next_question,
        inputs=[state],
        outputs=OUTPUTS,
    )

    restart_btn.click(
        fn=restart,
        inputs=[state],
        outputs=OUTPUTS,
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
