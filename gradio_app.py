"""Gradio 前端：MockInterview 模拟面试系统。

启动：
    python gradio_app.py
访问：http://localhost:7862
"""

import asyncio
import os
import sys
import threading
import time
import concurrent.futures
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from interview.runner import _ask, _answer_ai, _evaluate, _generate_report
from interview.models import InterviewSession
from agents import create_interviewer, create_interviewee, create_evaluator
from utils import build_interview_context, build_history_summary
from config import interview as interview_cfg
from config import model as model_cfg


# ─── Design System ─────────────────────────────────────────────────────────────
# Font : Apple SF Pro (system font, 无需加载)
# Color: Warm sand — #EDEAE5 bg · #F7F5F1 cards · #1A1714 text

CUSTOM_CSS = """
/* ── Global ──────────────────────────────────────────────────────────────── */
*, *::before, *::after {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text',
                 'Helvetica Neue', Arial, sans-serif !important;
    box-sizing: border-box;
    -webkit-font-smoothing: antialiased !important;
}
body, .gradio-container { background: #EDEAE5 !important; }

/* ── Gradio 外层清零 ─────────────────────────────────────────────────────── */
.gradio-container { padding: 0 !important; }
.gradio-container > .main { padding: 0 !important; }
.gradio-container > .main > .wrap { padding: 0 !important; }
.block { min-height: unset !important; }
.gap, .form { gap: 10px !important; }

/* ── App wrapper ─────────────────────────────────────────────────────────── */
#app-root { max-width: 860px; margin: 0 auto; padding: 0 24px 40px; }

/* ── Header ──────────────────────────────────────────────────────────────── */
#app-header {
    text-align: center !important;
    padding: 36px 0 24px !important;
    min-height: unset !important;
}
#app-header .prose { margin: 0 !important; }
#app-header .prose p {
    margin: 0 !important;
    line-height: 1 !important;
    color: #1A1714 !important;
}
#app-header .prose strong {
    font-size: 32px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    color: #1A1714 !important;
    display: block !important;
    margin-bottom: 8px !important;
}
#app-header .prose em {
    font-size: 14px !important;
    font-style: normal !important;
    color: #8C7E74 !important;
    letter-spacing: 0.02em !important;
}

/* ── Status text ─────────────────────────────────────────────────────────── */
#status-md .prose p {
    font-size: 13px !important;
    color: #9C8F84 !important;
    font-style: italic !important;
    margin: 6px 0 0 !important;
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
#setup-col, #interview-row, #eval-col, #report-col {
    background: #F7F5F1 !important;
    border-radius: 16px !important;
    border: 1px solid #D9D4CC !important;
    box-shadow: 0 2px 12px rgba(26,23,20,0.07) !important;
    margin-bottom: 0 !important;
    overflow: visible !important;
}
#setup-col    { padding: 24px 28px 20px !important; }
#interview-row { padding: 22px 24px !important; }
#eval-col {
    border-top: 3px solid #B5A99E !important;
    padding: 22px 24px !important;
}
/* ── Report card ─────────────────────────────────────────────────────────── */
#report-col {
    border-top: 4px solid #7C6357 !important;
    padding: 24px 28px !important;
    overflow: visible !important;
}

/* ── Section labels ──────────────────────────────────────────────────────── */
.section-header .prose p {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #9C8F84 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 0 0 14px !important;
}

/* ── Progress badge ──────────────────────────────────────────────────────── */
#progress-badge .prose { display: inline-block !important; }
#progress-badge .prose strong {
    background: #1A1714 !important;
    color: #F7F5F1 !important;
    border-radius: 20px !important;
    padding: 3px 14px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    display: inline-block !important;
}

/* ── Inner cards（question / eval）──────────────────────────────────────── */
#question-card, #eval-inner-card {
    background: #EAE7E2 !important;
    border-radius: 12px !important;
    padding: 16px 18px 20px !important;
    margin: 8px 0 14px !important;
    border: 1px solid #D9D4CC !important;
    overflow: visible !important;
}
#question-card .prose p {
    font-size: 16px !important;
    color: #1A1714 !important;
    line-height: 1.7 !important;
    margin: 0 !important;
}

/* ── Textareas ───────────────────────────────────────────────────────────── */
textarea, input[type="text"] {
    background: #EAE7E2 !important;
    border: 1.5px solid #D9D4CC !important;
    border-radius: 10px !important;
    color: #1A1714 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 10px 12px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    resize: vertical !important;
}
/* 限制 setup 区文本框高度，防止简历内容把页面撑爆 */
#setup-col textarea {
    max-height: 180px !important;
    overflow-y: auto !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #8C7E74 !important;
    box-shadow: 0 0 0 3px rgba(140,126,116,0.18) !important;
    background: #F7F5F1 !important;
    outline: none !important;
}
textarea:disabled {
    background: #E0DDD8 !important;
    color: #9C8F84 !important;
    cursor: not-allowed !important;
}

/* ── Labels ──────────────────────────────────────────────────────────────── */
label > span, .label-wrap > span {
    font-weight: 500 !important;
    font-size: 13px !important;
    color: #5C524C !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    line-height: 1.5 !important;
    margin-bottom: 4px !important;
}

/* ── Dropdown ────────────────────────────────────────────────────────────── */
.wrap-inner {
    border-radius: 10px !important;
    border: 1.5px solid #D9D4CC !important;
    background: #EAE7E2 !important;
    font-size: 14px !important;
}

/* ── File upload ─────────────────────────────────────────────────────────── */
[data-testid="file"] {
    border-radius: 10px !important;
    border: 1.5px dashed #C8C2B8 !important;
    background: #E8E4DF !important;
    min-height: 52px !important;
    height: 52px !important;
    transition: border-color 0.15s !important;
}
[data-testid="file"] > div {
    padding: 0 14px !important; height: 52px !important;
    justify-content: center !important;
}
[data-testid="file"]:hover { border-color: #8C7E74 !important; }
[data-testid="file"] span, [data-testid="file"] p {
    font-size: 13px !important; color: #7A6B62 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
button.lg.primary, button.primary {
    background: #1A1714 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    color: #F7F5F1 !important;
    box-shadow: 0 2px 8px rgba(26,23,20,0.2) !important;
    transition: opacity 0.15s !important;
    cursor: pointer !important;
}
button.lg.primary:hover, button.primary:hover { opacity: 0.8 !important; }
button.lg.primary:disabled, button.primary:disabled {
    background: #B8B0A8 !important; cursor: not-allowed !important; opacity: 1 !important;
}
button.lg.secondary, button.secondary {
    background: #E0DDD8 !important;
    border: 1.5px solid #D9D4CC !important;
    border-radius: 10px !important;
    color: #1A1714 !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    transition: background 0.15s !important;
    cursor: pointer !important;
}
button.lg.secondary:hover, button.secondary:hover { background: #D4CFC8 !important; }

/* ── Setup controls ──────────────────────────────────────────────────────── */
#setup-controls { align-items: flex-end !important; gap: 12px !important; }

/* ── Eval score ──────────────────────────────────────────────────────────── */
#eval-md .prose h2 {
    font-size: 28px !important; font-weight: 700 !important;
    color: #1A1714 !important; letter-spacing: -0.5px !important;
    margin-bottom: 6px !important;
}
#eval-md .prose p  { font-size: 14px !important; color: #3A342F !important; line-height: 1.65 !important; }
#eval-md .prose li { font-size: 14px !important; color: #3A342F !important; }
#eval-md .prose strong { color: #1A1714 !important; font-weight: 600 !important; }

/* ── Report ──────────────────────────────────────────────────────────────── */
#report-md .prose h2 {
    font-size: 52px !important;
    font-weight: 800 !important;
    color: #1A1714 !important;
    letter-spacing: -2.5px !important;
    line-height: 1 !important;
    margin: 0 0 12px !important;
}
#report-md .prose h3 {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #8C7E74 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    margin: 0 0 10px !important;
}
#report-md .prose p { font-size: 14px !important; color: #3A342F !important; line-height: 1.65 !important; margin: 0 0 4px !important; }
#report-md .prose li { font-size: 14px !important; color: #3A342F !important; line-height: 1.7 !important; }
#report-md .prose hr { border: none !important; border-top: 1px solid #D9D4CC !important; margin: 18px 0 !important; }

/* ── Prose defaults ──────────────────────────────────────────────────────── */
.prose p    { color: #3A342F !important; line-height: 1.7 !important; font-size: 14px !important; }
.prose li   { color: #3A342F !important; }
.prose strong { color: #1A1714 !important; }

/* ── 禁用 Gradio 加载时的边框脉冲动画，只保留蒙板 ────────────────────────── */
.generating { animation: none !important; border-color: inherit !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #C8C2B8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9C8F84; }

/* ── Mobile ──────────────────────────────────────────────────────────────── */
@media (max-width: 700px) {
    #app-root { padding: 0 14px 28px; }
}
"""


# ─── Theme ────────────────────────────────────────────────────────────────────

def _make_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.stone,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        font=["system-ui", "-apple-system", "Helvetica Neue", "sans-serif"],
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_md,
    ).set(
        body_background_fill="#EDEAE5",
        body_text_color="#1A1714",
        border_color_primary="#D9D4CC",
        background_fill_primary="#F7F5F1",
        background_fill_secondary="#EAE7E2",
        shadow_drop="0 2px 12px rgba(26,23,20,0.07)",
        shadow_drop_lg="0 6px 24px rgba(26,23,20,0.1)",
        input_background_fill="#EAE7E2",
        input_border_color="#D9D4CC",
        input_border_color_focus="#8C7E74",
        input_border_color_hover="#C4BDB5",
        button_primary_background_fill="#1A1714",
        button_primary_background_fill_hover="#352E29",
        button_primary_text_color="#F7F5F1",
        button_secondary_background_fill="#E0DDD8",
        button_secondary_background_fill_hover="#D4CFC8",
        button_secondary_text_color="#1A1714",
        button_secondary_border_color="#D9D4CC",
        block_radius="16px",
        button_large_radius="10px",
        button_small_radius="8px",
        input_radius="10px",
        block_border_width="1px",
        block_border_color="#D9D4CC",
        block_background_fill="#F7F5F1",
        block_shadow="0 2px 12px rgba(26,23,20,0.07)",
    )


# ─── RAG 异步生命周期 ──────────────────────────────────────────────────────────
# MCPTools 是 async context manager，需要一个常驻的 event loop 让 subprocess 保持连接。
# 使用守护线程持有 loop（run_forever），通过 run_coroutine_threadsafe 与 Gradio 同步代码桥接。

_rag_loop: Optional[asyncio.AbstractEventLoop] = None


def _ensure_rag_loop() -> asyncio.AbstractEventLoop:
    global _rag_loop
    if _rag_loop is None or _rag_loop.is_closed():
        _rag_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_rag_loop.run_forever, daemon=True)
        t.start()
    return _rag_loop


def _start_rag(timeout_s: int = 30):
    """启动 MCPTools subprocess，返回 mcp_instance。

    agno manual 模式：
      await mcp_tools.connect()  → 建立连接（返回 None，不是工具对象）
      agent = Agent(tools=[mcp_tools])  → mcp_instance 本身即工具
      await mcp_tools.close()   → 关闭
    """
    from agno.tools.mcp import MCPTools
    from config import mcp as mcp_cfg
    loop = _ensure_rag_loop()
    mcp_instance = MCPTools(
        command=mcp_cfg.command,
        transport="stdio",
        include_tools=["query_knowledge_hub", "list_collections"],
    )
    fut = asyncio.run_coroutine_threadsafe(mcp_instance.connect(), loop)
    fut.result(timeout=timeout_s)   # connect() 返回 None，等待连接建立即可
    return mcp_instance             # mcp_instance 本身就是传给 Agent 的工具


def _stop_rag(mcp_instance) -> None:
    """关闭 MCPTools subprocess。"""
    if mcp_instance is None:
        return
    try:
        loop = _ensure_rag_loop()
        fut = asyncio.run_coroutine_threadsafe(mcp_instance.close(), loop)
        fut.result(timeout=10)
    except Exception as e:
        print(f"[WARN] RAG 关闭异常：{e}", flush=True)


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────

AGENT_CALL_TIMEOUT_S = int(os.getenv("AGENT_CALL_TIMEOUT_S", "60"))
MODEL_HEALTHCHECK_TIMEOUT_S = float(os.getenv("MODEL_HEALTHCHECK_TIMEOUT_S", "2.5"))


def _check_model_server() -> None:
    """快速检查 vLLM(OpenAI-compatible) 服务是否可用，避免前端长时间卡住。"""
    base = (model_cfg.base_url or "").rstrip("/")
    if not base:
        raise gr.Error("模型服务地址为空：请设置 VLLM_BASE_URL。")

    url = f"{base}/models"  # vLLM: /v1/models
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=MODEL_HEALTHCHECK_TIMEOUT_S) as resp:
            status = getattr(resp, "status", 200)
            if not (200 <= status < 300):
                raise gr.Error(f"模型服务不可用：{url} 返回状态码 {status}")
    except (HTTPError, URLError, TimeoutError) as e:
        raise gr.Error(
            "模型服务不可用或响应超时。\n"
            f"- 检查地址：{url}\n"
            f"- 解决：确认 vLLM 已启动且可访问（例如监听 localhost:8000），或调整环境变量 VLLM_BASE_URL。\n"
            f"- 错误：{e}"
        )


def _call_with_timeout(desc: str, fn, *args, timeout_s: int = AGENT_CALL_TIMEOUT_S, **fn_kwargs):
    """为阻塞式模型调用增加超时，避免 Gradio 会话一直处于 Running。

    注意：不使用 with 语句，避免 ThreadPoolExecutor.__exit__ 调用
    shutdown(wait=True) 时永久阻塞（线程挂起时永远不会结束）。
    fn_kwargs 会透传给 fn（timeout_s 本身不会传入）。
    """
    start = time.time()
    print(f"[INFO] {desc} 开始（timeout={timeout_s}s）", flush=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(fn, *args, **fn_kwargs)
    try:
        result = fut.result(timeout=timeout_s)
        cost = time.time() - start
        print(f"[INFO] {desc} 结束（{cost:.2f}s）", flush=True)
        return result
    except concurrent.futures.TimeoutError as e:
        cost = time.time() - start
        print(f"[WARN] {desc} 超时（{cost:.2f}s），放弃等待线程", flush=True)
        raise TimeoutError(f"{desc} 超时（>{timeout_s}s）") from e
    finally:
        # wait=False：不阻塞等待挂起的线程，让其在后台自然结束
        executor.shutdown(wait=False)


def _resolve_file_path(file_obj) -> str:
    """从 Gradio 文件对象中提取实际路径（兼容 str / NamedString / dict）。"""
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path", "")
    return str(file_obj)


def _read_uploaded_file(file_obj) -> str:
    """读取上传的文件内容，支持 .txt 和 .pdf。"""
    path = _resolve_file_path(file_obj)
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            import fitz  # pymupdf
        except ImportError:
            raise gr.Error("pymupdf 未安装，请运行 pip install pymupdf")
        doc = fitz.open(str(p))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip():
            raise gr.Error(
                "PDF 未能提取到文字。\n"
                "可能原因：使用了图片扫描或特殊字体编码。\n"
                "建议：将简历另存为 .txt 文件后上传，或直接粘贴文字。"
            )
        return text.strip()
    return p.read_text(encoding="utf-8")


def _is_last_question(state: dict) -> bool:
    at_last_main = state["current_q_idx"] >= state["num_questions"]
    no_more_followup = (
        state["is_follow_up"]
        or state["num_follow_ups"] == 0
        or state["follow_up_count"] >= state["num_follow_ups"]
    )
    return at_last_main and no_more_followup


def _format_eval(turn) -> str:
    score = turn.score
    if score >= 8:
        label = "优秀"
    elif score >= 6:
        label = "良好"
    else:
        label = "待加强"
    lines = [f"## {score:.1f} / 10 · {label}", "", turn.feedback or ""]
    if turn.strengths:
        lines += ["", "**优点**"]
        lines += [f"- {s}" for s in turn.strengths]
    if turn.areas_for_improvement:
        lines += ["", "**改进方向**"]
        lines += [f"- {s}" for s in turn.areas_for_improvement]
    return "\n".join(lines)


def _format_report(report) -> str:
    rec_map = {
        "Strong Hire": "强烈推荐录用",
        "Hire":        "推荐录用",
        "Maybe":       "待定",
        "No Hire":     "暂不录用",
    }
    rec_label = rec_map.get(report.recommendation, report.recommendation)

    lines = [
        f"## {report.overall_score:.0f} / 100",
        "",
        f"**{report.candidate_name}** · {report.position} · {rec_label}",
        "",
        "---",
        "",
        f"技术能力 **{report.technical_score:.1f}** / 10  ·  沟通表达 **{report.communication_score:.1f}** / 10",
    ]
    if report.key_strengths:
        lines += ["", "---", "", "### 核心优势", ""]
        lines += [f"- {s}" for s in report.key_strengths]
    if report.skill_gaps:
        lines += ["", "---", "", "### 技能缺口", ""]
        lines += [f"- {s}" for s in report.skill_gaps]
    if report.improvement_suggestions:
        lines += ["", "---", "", "### 改进建议", ""]
        lines += [f"- {s}" for s in report.improvement_suggestions]
    return "\n".join(lines)


# ─── State ────────────────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "phase": "setup",
        "session": None,
        "interviewer": None,
        "evaluator": None,
        "interviewee": None,
        "mcp_instance": None,
        "mode": "human",
        "use_rag": False,
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


# ─── outputs 构造器 ────────────────────────────────────────────────────────────
# 顺序: state, setup_col, interview_col, progress_md, question_md,
#       answer_box, submit_btn, status_md, eval_col, eval_md, next_btn,
#       report_col, report_md

def _outputs(
    state,
    setup_visible=None,
    interview_visible=None,
    progress=None,
    question=None,
    answer_value=None,
    answer_interactive=None,
    submit_interactive=None,
    submit_value=None,
    status_value=None,
    eval_col_visible=None,
    eval_value=None,
    next_value=None,
    next_visible=None,          # 独立控制 next_btn 可见性（不跟随 eval_col_visible）
    report_col_visible=None,
    report_value=None,
):
    def _vis(val):
        return gr.update() if val is None else gr.update(visible=val)

    def _v(val):
        return gr.update() if val is None else gr.update(value=val)

    answer_upd = gr.update()
    if answer_value is not None and answer_interactive is not None:
        answer_upd = gr.update(value=answer_value, interactive=answer_interactive)
    elif answer_value is not None:
        answer_upd = gr.update(value=answer_value)
    elif answer_interactive is not None:
        answer_upd = gr.update(interactive=answer_interactive)

    # submit_btn：可同时更新 interactive + value（按钮文字）
    if submit_interactive is not None and submit_value is not None:
        submit_upd = gr.update(interactive=submit_interactive, value=submit_value)
    elif submit_interactive is not None:
        submit_upd = gr.update(interactive=submit_interactive)
    elif submit_value is not None:
        submit_upd = gr.update(value=submit_value)
    else:
        submit_upd = gr.update()

    # next_btn：next_visible 优先；其次跟随 eval_col_visible；最后只更新 value。
    if next_visible is not None:
        next_btn_upd = (
            gr.update(visible=next_visible, value=next_value)
            if next_value is not None
            else gr.update(visible=next_visible)
        )
    elif eval_col_visible is not None and next_value is not None:
        next_btn_upd = gr.update(visible=eval_col_visible, value=next_value)
    elif eval_col_visible is not None:
        next_btn_upd = gr.update(visible=eval_col_visible)
    elif next_value is not None:
        next_btn_upd = gr.update(value=next_value)
    else:
        next_btn_upd = gr.update()

    return (
        state,
        _vis(setup_visible),
        _vis(interview_visible),
        _v(progress),
        _v(question),
        answer_upd,
        submit_upd,
        _v(status_value),
        _vis(eval_col_visible),
        _v(eval_value),
        next_btn_upd,
        _vis(report_col_visible),
        _v(report_value),
    )


# ─── 事件处理函数 ──────────────────────────────────────────────────────────────

def on_resume_upload(file_obj, current_text: str):
    if file_obj is None:
        return gr.update()
    try:
        text = _read_uploaded_file(file_obj)
        return gr.update(value=text)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"文件读取失败：{e}")


def start_interview(resume: str, jd: str, num_q: int, num_fu: int,
                    mode_str: str, use_rag: bool, state: dict):
    """普通函数（非 generator），返回第一题的 UI 状态。
    loading 提示由 start_btn.click 预步骤 lambda 负责显示。
    """
    import traceback

    # 先清理上一次可能留下的 RAG 连接
    if state and state.get("mcp_instance"):
        _stop_rag(state["mcp_instance"])

    try:
        if not resume.strip() or not jd.strip():
            gr.Warning("请填写简历和职位描述后再开始。")
            return _outputs(state, status_value="")

        _check_model_server()

        num_q   = int(num_q)
        num_fu  = int(num_fu)
        mode    = "ai" if mode_str == "AI 模拟" else "human"
        candidate_name = resume.strip().splitlines()[0][:50]
        position       = jd.strip().splitlines()[0][:50]

        session = InterviewSession(
            resume_text=resume,
            jd_text=jd,
            candidate_name=candidate_name,
            position=position,
            total_rounds=num_q,
        )

        # RAG：启动 MCPTools subprocess，mcp_instance 本身即传给 Agent 的工具
        mcp_instance = None
        if use_rag:
            print("[INFO] 正在连接 RAG MCP 服务器...", flush=True)
            mcp_instance = _call_with_timeout(
                "启动 RAG", _start_rag, timeout_s=40,
            )
            print("[INFO] RAG 连接成功", flush=True)

        interviewer = create_interviewer(rag_tools=[mcp_instance] if mcp_instance else None)
        evaluator   = create_evaluator()
        interviewee = create_interviewee(resume_context=resume) if mode == "ai" else None
        context     = build_interview_context(resume, jd)

        state = _empty_state()
        state.update({
            "phase": "questioning",
            "session": session,
            "interviewer": interviewer,
            "evaluator": evaluator,
            "interviewee": interviewee,
            "mcp_instance": mcp_instance,
            "mode": mode,
            "use_rag": use_rag,
            "context": context,
            "current_q_idx": 1,
            "num_questions": num_q,
            "num_follow_ups": num_fu,
        })

        rag_loop = _rag_loop if use_rag else None
        question, rag_hit = _call_with_timeout(
            "生成第 1 题", _ask,
            interviewer, context, "", 1,
            _mcp=mcp_instance, _loop=rag_loop,
        )
        question = (question or "").strip() or "（未生成到问题文本，请点击「重新开始」后重试）"
        state["current_question"] = question
        print(f"[INTERVIEWER Q1] {question}", flush=True)

        ai_mode = (mode == "ai")
        rag_status = "✓ 面试官已检索题库" if rag_hit else ("⚠ 未检索题库（模型直接出题）" if state.get("use_rag") else "")
        return _outputs(
            state,
            setup_visible=False,
            interview_visible=True,
            progress=f"**第 1 / {num_q} 题**",
            question=question,
            answer_value="",
            answer_interactive=not ai_mode,
            submit_interactive=True,
            submit_value="让 AI 回答" if ai_mode else "提交回答",
            status_value=rag_status,
            eval_value="",
            report_value="",
        )
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[ERROR] start_interview 异常:\n{tb}", flush=True)
        _stop_rag(mcp_instance if 'mcp_instance' in dir() else None)
        return _outputs(
            _empty_state(),
            setup_visible=True,
            status_value=f"❌ 启动失败：{exc}（详情见终端）",
        )


def submit_answer(answer: str, state: dict):
    import traceback
    try:
        if state is None:
            return _outputs(_empty_state(), submit_interactive=True, status_value="",
                            eval_col_visible=True,
                            eval_value="⚠️ 状态丢失，请刷新页面重新开始。")

        phase = state.get("phase")
        if phase != "questioning":
            return _outputs(state, submit_interactive=True, status_value="",
                            eval_col_visible=True,
                            eval_value=f"⚠️ 当前阶段为 {phase!r}，不在答题中。请先点击「开始面试」。")

        ai_mode = (state.get("mode") == "ai")

        if ai_mode:
            # AI 模式：由 IntervieweeAgent 生成回答，忽略 answer 输入框
            interviewee = state["interviewee"]
            answer = _call_with_timeout(
                "AI 生成回答", _answer_ai,
                interviewee, state["current_question"],
            )
            answer = (answer or "").strip() or "（AI 未生成到回答）"
            print(f"[INTERVIEWEE Q{state['current_q_idx']}] {answer[:80]}...", flush=True)
        else:
            answer = answer.strip()
            if not answer:
                return _outputs(state, submit_interactive=True, status_value="",
                                eval_col_visible=True,
                                eval_value="⚠️ 请输入回答后再提交。")

        turn = _call_with_timeout(
            "评估本题", _evaluate,
            state["evaluator"], state["current_question"], answer,
        )
        turn.question_type = "follow_up" if state["is_follow_up"] else "main"
        state["session"].turns.append(turn)
        state["last_answer"]   = answer
        state["last_score"]    = turn.score
        state["last_feedback"] = turn.feedback
        state["phase"]         = "evaluating"

        q_idx = state["current_q_idx"]
        suffix = "↳" if turn.question_type == "follow_up" else ""
        print(f"[EVAL Q{q_idx}{suffix}] score={turn.score:.1f} | {turn.feedback[:60]}...", flush=True)

        next_label = "生成报告" if _is_last_question(state) else "下一题"
        return _outputs(
            state,
            answer_value=answer if ai_mode else None,   # AI 模式：回填答案文本
            answer_interactive=False,
            submit_interactive=False,
            status_value="",
            eval_col_visible=True,
            eval_value=_format_eval(turn),
            next_value=next_label,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[ERROR] submit_answer 异常:\n{tb}", flush=True)
        return _outputs(
            state or _empty_state(),
            submit_interactive=True,
            status_value="",
            eval_col_visible=True,
            eval_value=f"❌ 出错：{exc}\n\n（详情见终端日志）",
        )


def next_question(state: dict):
    """Generator 函数：支持报告路径的两阶段 yield，避免 Gradio 在隐藏触发按钮时
    无法正常关闭 loading 状态的问题。
    非报告路径只 yield 一次，行为与普通函数等价。
    """
    import traceback
    try:
        if state is None or state.get("phase") != "evaluating":
            yield _outputs(state, status_value="")
            return

        _check_model_server()

        session     = state["session"]
        interviewer = state["interviewer"]
        evaluator   = state["evaluator"]

        ai_mode  = (state.get("mode") == "ai")
        rag_loop = _rag_loop if state.get("use_rag") else None

        # 追问
        can_follow_up = (
            not state["is_follow_up"]
            and state["num_follow_ups"] > 0
            and state["follow_up_count"] < state["num_follow_ups"]
        )
        if can_follow_up:
            fu_q, _ = _call_with_timeout(
                f"生成追问（第 {state['current_q_idx']} 题）",
                _ask,
                interviewer, state["context"], "",
                state["current_q_idx"],
                True,
                state["last_answer"], state["last_score"], state["last_feedback"],
                _mcp=state.get("mcp_instance"), _loop=rag_loop,
            )
            fu_q = (fu_q or "").strip() or "（未生成到追问文本，请重试）"
            state["current_question"]  = fu_q
            state["is_follow_up"]      = True
            state["follow_up_count"]  += 1
            state["phase"]             = "questioning"

            q_idx, num_q = state["current_q_idx"], state["num_questions"]
            print(f"[INTERVIEWER Q{q_idx}↳] {fu_q}", flush=True)
            yield _outputs(
                state,
                progress=f"**第 {q_idx} / {num_q} 题（追问）**",
                question=fu_q,
                answer_value="",
                answer_interactive=not ai_mode,
                submit_interactive=True,
                submit_value="让 AI 回答" if ai_mode else "提交回答",
                status_value="",
                eval_col_visible=False,
            )
            return

        # 下一主题
        if state["current_q_idx"] < state["num_questions"]:
            state["current_q_idx"]  += 1
            state["is_follow_up"]    = False
            state["follow_up_count"] = 0
            state["phase"]           = "questioning"

            history  = build_history_summary(session.turns, last_n=2)
            question, rag_hit = _call_with_timeout(
                f"生成第 {state['current_q_idx']} 题",
                _ask,
                interviewer, state["context"], history, state["current_q_idx"],
                _mcp=state.get("mcp_instance"), _loop=rag_loop,
            )
            question = (question or "").strip() or "（未生成到问题文本，请重试）"
            state["current_question"] = question

            q_idx, num_q = state["current_q_idx"], state["num_questions"]
            print(f"[INTERVIEWER Q{q_idx}] {question}", flush=True)
            rag_status = "✓ 面试官已检索题库" if rag_hit else ("⚠ 未检索题库（模型直接出题）" if state.get("use_rag") else "")
            yield _outputs(
                state,
                progress=f"**第 {q_idx} / {num_q} 题**",
                question=question,
                answer_value="",
                answer_interactive=not ai_mode,
                submit_interactive=True,
                submit_value="让 AI 回答" if ai_mode else "提交回答",
                status_value=rag_status,
                eval_col_visible=False,
            )
            return

        # ── 生成报告：两阶段 yield ────────────────────────────────────────────
        # 第一次 yield：隐藏 interview_row / eval_col，显示 report_col 加载态。
        # 注意：next_btn 是 generator 的触发元素，不能在此隐藏——隐藏会导致 Gradio
        # 的 loading overlay 失去锚点，表现为幽灵卡片与 report_col 重叠。
        # 改为保留 next_btn 可见（显示"生成中..."），第二次 yield 再隐藏。
        state["phase"] = "report"
        n_turns = len(session.turns)
        yield _outputs(
            state,
            interview_visible=False,
            eval_col_visible=False,
            next_visible=True,          # 保留触发按钮可见，防止 loading overlay 幽灵卡片
            next_value="⏳ 生成中...",
            report_col_visible=True,
            report_value=f"*正在分析 {n_turns} 轮面试对话，生成综合评估...*",
            status_value="",
        )

        report = _call_with_timeout("生成最终报告", _generate_report, evaluator, session)

        # 第二次 yield：写入正式报告，此时隐藏 next_btn。
        yield _outputs(
            state,
            next_visible=False,
            report_value=_format_report(report),
        )

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[ERROR] next_question 异常:\n{tb}", flush=True)
        if state:
            state["phase"] = "evaluating"
        yield _outputs(
            state or _empty_state(),
            eval_col_visible=True,
            eval_value=f"❌ 出错：{exc}\n\n（详情见终端日志）",
            next_value="重试",
            status_value="",
        )


def restart(state: dict):
    if state and state.get("mcp_instance"):
        _stop_rag(state["mcp_instance"])
    return _outputs(
        _empty_state(),
        setup_visible=True,
        interview_visible=False,
        progress="",
        question="",
        answer_value="",
        answer_interactive=True,
        submit_interactive=True,
        submit_value="提交回答",
        eval_col_visible=False,
        eval_value="",
        next_value="下一题",
        report_col_visible=False,
        report_value="",
    )


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="MockInterview",
    analytics_enabled=False,
) as demo:

    with gr.Column(elem_id="app-root"):

        # ── Header ───────────────────────────────────────────────────────────
        gr.Markdown(
            "**🎤 MockInterview**\n\n*模拟面试 · 即时评估 · 生成报告*",
            elem_id="app-header",
        )

        state     = gr.State(_empty_state())
        status_md = gr.Markdown("", elem_id="status-md")

        # ── Setup 区 ──────────────────────────────────────────────────────────
        with gr.Column(visible=True, elem_id="setup-col") as setup_col:
            with gr.Column(elem_classes="section-header"):
                gr.Markdown("📋 配置面试")

            resume_upload = gr.File(
                label="📄 上传简历（.txt / .pdf）",
                file_types=[".txt", ".pdf"],
                file_count="single",
            )
            resume_box = gr.Textbox(
                label="简历内容（第一行为候选人姓名）",
                lines=6,
                placeholder="张三\n5年 Python 后端开发经验\n...\n\n上传后自动填入，也可直接粘贴。",
            )
            jd_box = gr.Textbox(
                label="职位描述（第一行为职位名称）",
                lines=5,
                placeholder="高级后端工程师\n负责核心服务开发...\n要求：Python · Go · 分布式系统",
            )

            with gr.Row():
                mode_radio = gr.Radio(
                    choices=["人工练习", "AI 模拟"],
                    value="人工练习",
                    label="面试模式",
                )
                rag_checkbox = gr.Checkbox(
                    label="启用 RAG 题库检索",
                    value=False,
                )

            # 配置项 + 开始按钮同一行
            with gr.Row(elem_id="setup-controls"):
                num_q_dropdown = gr.Dropdown(
                    choices=[3, 4, 5, 6, 7, 8],
                    value=interview_cfg.num_questions,
                    label="面试题数",
                )
                num_fu_dropdown = gr.Dropdown(
                    choices=[0, 1, 2],
                    value=interview_cfg.max_follow_ups,
                    label="每题追问次数",
                )
                start_btn = gr.Button("🚀 开始面试", variant="primary", size="lg")

        # ── Interview 区（问题+回答）──────────────────────────────────────────
        with gr.Column(visible=False, elem_id="interview-row") as interview_row:
            progress_md = gr.Markdown("**第 1 / 5 题**", elem_id="progress-badge")

            with gr.Column(elem_id="question-card"):
                gr.Markdown("**面试官**")
                question_md = gr.Markdown("")

            answer_box = gr.Textbox(
                label="你的回答",
                lines=3,
                placeholder="在此输入回答...",
                interactive=True,
            )
            submit_btn = gr.Button("提交回答", variant="primary")

        # ── 评分区（顶层，visible 通过 gr.update 切换）───────────────────────
        with gr.Column(visible=False, elem_id="eval-col") as eval_col:
            with gr.Column(elem_id="eval-inner-card"):
                gr.Markdown("**评分**")
                eval_md = gr.Markdown("", elem_id="eval-md")

        # next_btn 独立于 eval_col 之外，避免点击事件触发时按钮容器被同时隐藏
        # 导致 Gradio 前端 loading 状态无法正常关闭。
        next_btn = gr.Button("下一题", variant="secondary", visible=False)

        # ── Report 区（顶层，visible 通过 gr.update 切换）────────────────────
        with gr.Column(visible=False, elem_id="report-col") as report_col:
            report_md = gr.Markdown("", elem_id="report-md")
            restart_btn = gr.Button("重新开始", variant="secondary")

        # ── outputs 列表（13 项）──────────────────────────────────────────────
        OUTPUTS = [
            state,
            setup_col, interview_row,
            progress_md, question_md,
            answer_box, submit_btn, status_md,
            eval_col, eval_md, next_btn,
            report_col, report_md,
        ]

        # ── 事件绑定 ──────────────────────────────────────────────────────────
        resume_upload.upload(
            fn=on_resume_upload,
            inputs=[resume_upload, resume_box],
            outputs=[resume_box],
        )

        # 开始面试：预步骤禁用按钮+显示 loading 文字（queue=False 立即生效）
        # start_interview 已改为普通函数，避免 generator+then 链导致 loading 不结束
        start_btn.click(
            fn=lambda: (gr.update(interactive=False),
                        gr.update(value="⏳ 正在生成第 1 题，请稍候...")),
            inputs=[],
            outputs=[start_btn, status_md],
            queue=False,
        ).then(
            fn=start_interview,
            inputs=[resume_box, jd_box, num_q_dropdown, num_fu_dropdown,
                    mode_radio, rag_checkbox, state],
            outputs=OUTPUTS,
        ).then(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[start_btn],
            queue=False,
        )

        # 提交回答：先在 status_md 显示"正在评估"（不动卡片边框），再跑评估
        submit_btn.click(
            fn=lambda: (gr.update(interactive=False), gr.update(value="⏳ 正在处理，请稍候...")),
            inputs=[],
            outputs=[submit_btn, status_md],
            queue=False,
        ).then(
            fn=submit_answer,
            inputs=[answer_box, state],
            outputs=OUTPUTS,
        )

        # 下一题 / 生成报告：先在 status_md 显示"正在生成"，再跑
        next_btn.click(
            fn=lambda: gr.update(value="⏳ 正在生成，请稍候..."),
            inputs=[],
            outputs=[status_md],
            queue=False,
        ).then(
            fn=next_question,
            inputs=[state],
            outputs=OUTPUTS,
        )

        # 重新开始
        restart_btn.click(
            fn=restart,
            inputs=[state],
            outputs=OUTPUTS,
        )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False,
                theme=_make_theme(), css=CUSTOM_CSS)
