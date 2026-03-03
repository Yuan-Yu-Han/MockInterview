"""Gradio web demo for Qwen3-ASR-0.6B — click-to-record with auto transcribe."""
import torch
import gradio as gr
from qwen_asr import Qwen3ASRModel

MODEL_PATH = "/projects/yuan0165/Qwen3-ASR-0.6B"

LANGUAGES = [
    "Auto Detect",
    "Chinese", "English", "Cantonese", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Italian",
    "Russian", "Arabic", "Hindi", "Thai", "Vietnamese",
    "Indonesian", "Malay", "Dutch", "Turkish",
]

print(f"Loading {MODEL_PATH} ...")
model = Qwen3ASRModel.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=4,
    max_new_tokens=512,
)
print("Model ready.\n")


def transcribe(audio_path: str, language: str) -> tuple[str, str]:
    if audio_path is None:
        return "", ""
    lang = None if language == "Auto Detect" else language
    results = model.transcribe(audio=audio_path, language=lang)
    r = results[0]
    return r.language or "unknown", r.text or ""


with gr.Blocks(title="Qwen3-ASR Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Qwen3-ASR-0.6B 语音识别")
    gr.Markdown("点击麦克风开始录音，停止后自动识别。也可以上传音频文件。")

    with gr.Row():
        # ── 左栏：输入 ──────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 输入")

            mic_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 点击录音（停止后自动识别）",
            )

            with gr.Accordion("📂 或上传音频文件", open=False):
                file_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="上传 WAV / MP3 / FLAC",
                )
                upload_btn = gr.Button("识别上传的文件", variant="secondary")

            lang_dropdown = gr.Dropdown(
                choices=LANGUAGES,
                value="Auto Detect",
                label="语言",
            )

        # ── 右栏：输出 ──────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 识别结果")
            lang_out = gr.Textbox(
                label="检测到的语言",
                interactive=False,
                max_lines=1,
            )
            text_out = gr.Textbox(
                label="转录文本",
                lines=10,
                interactive=False,
            )

    # 录音停止后自动触发
    mic_input.stop_recording(
        fn=transcribe,
        inputs=[mic_input, lang_dropdown],
        outputs=[lang_out, text_out],
    )

    # 上传文件手动点按钮
    upload_btn.click(
        fn=transcribe,
        inputs=[file_input, lang_dropdown],
        outputs=[lang_out, text_out],
    )

    gr.Examples(
        examples=[
            ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav", "Auto Detect"],
            ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav", "Auto Detect"],
        ],
        inputs=[file_input, lang_dropdown],
        fn=transcribe,
        outputs=[lang_out, text_out],
        cache_examples=False,
        label="示例音频（点击加载后点识别上传的文件按钮）",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
