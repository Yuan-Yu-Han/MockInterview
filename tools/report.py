"""Report serialization and export utilities."""

import json
from pathlib import Path
from datetime import datetime

from interview.models import InterviewReport


def save_report_json(report: InterviewReport, output_dir: Path) -> Path:
    """Save InterviewReport as a JSON file in output_dir.

    Filename pattern: report_<candidate>_<timestamp>.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = report.candidate_name.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{safe_name}_{timestamp}.json"
    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    return out_path


def save_report_text(report: InterviewReport, output_dir: Path) -> Path:
    """Save InterviewReport as a human-readable text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = report.candidate_name.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{safe_name}_{timestamp}.txt"
    out_path = output_dir / filename

    lines = [
        "=" * 70,
        "  MOCK INTERVIEW EVALUATION REPORT",
        "=" * 70,
        f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Candidate  : {report.candidate_name}",
        f"  Position   : {report.position}",
        "",
        "─── OVERALL ASSESSMENT ───────────────────────────────────────────────",
        f"  Recommendation  : {report.recommendation}",
        f"  Overall Score   : {report.overall_score:.1f} / 100",
        f"  Technical       : {report.technical_score:.1f} / 10",
        f"  Communication   : {report.communication_score:.1f} / 10",
        f"  Questions Asked : {report.questions_asked}",
        "",
        "─── KEY STRENGTHS ────────────────────────────────────────────────────",
    ]
    for s in report.key_strengths:
        lines.append(f"  • {s}")

    lines += [
        "",
        "─── SKILL GAPS ───────────────────────────────────────────────────────",
    ]
    for g in report.skill_gaps:
        lines.append(f"  • {g}")

    lines += [
        "",
        "─── IMPROVEMENT SUGGESTIONS ──────────────────────────────────────────",
    ]
    for s in report.improvement_suggestions:
        lines.append(f"  • {s}")

    lines += [
        "",
        "─── DETAILED Q&A FEEDBACK ────────────────────────────────────────────",
    ]
    for i, turn in enumerate(report.detailed_feedback, 1):
        kind = f"[{turn.question_type.upper()}]"
        lines += [
            f"\n  Turn {i} {kind}",
            f"  Q : {turn.question}",
            f"  A : {turn.answer}",
            f"  Score    : {turn.score:.1f}/10",
            f"  Feedback : {turn.feedback}",
        ]
        if turn.strengths:
            lines.append(f"  Strengths : {', '.join(turn.strengths)}")
        if turn.areas_for_improvement:
            lines.append(f"  Improve   : {', '.join(turn.areas_for_improvement)}")

    lines += ["", "=" * 70]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def make_save_tool(output_dir: Path, session):
    """Create a save_report closure for use as an agno agent tool.

    The closure captures output_dir and the session object.  When the agent
    calls save_report(report_json), the tool builds the full InterviewReport
    (adding session.turns as detailed_feedback) and writes JSON + TXT files.
    """
    from utils import safe_float, safe_list  # absolute import

    def save_report(report_json: str) -> str:
        """将最终评估报告保存为 JSON 和 TXT 文件。

        Args:
            report_json: 你刚刚生成的报告 JSON 字符串。
        """
        try:
            d = json.loads(report_json)
            avg = (
                sum(t.score for t in session.turns) / len(session.turns)
                if session.turns else 0.0
            )
            report = InterviewReport(
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
            json_path = save_report_json(report, output_dir)
            txt_path = save_report_text(report, output_dir)
            msg = f"报告已保存：\n  JSON: {json_path}\n  TXT:  {txt_path}"
            print(f"\n{msg}")
            return msg
        except Exception as exc:
            err = f"保存失败：{exc}"
            print(f"\n[警告] {err}")
            return err

    return save_report
