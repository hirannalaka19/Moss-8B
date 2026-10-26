"""Gradio web UI for MOSS-TTS-v1.5 (8B) long-form narration. Run: python app.py"""

import logging
import random
import re
import time
import uuid
from pathlib import Path

import gradio as gr

from moss_narrator.audio_utils import save_audio
from moss_narrator.engine import MossNarrator
from moss_narrator.planning import LANGUAGES, NarrationSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

OUTPUT_DIR = Path(__file__).resolve().parent / "generated_audio"
DEFAULTS = NarrationSettings()
PAUSE_TAGS = ["[pause 0.5s]", "[pause 1.0s]", "[pause 2.0s]", "[pause 3.0s]"]

# Inserts the clicked tag at the cursor position of the given textbox.
INSERT_TAG_JS = """
(tag_val, current_text) => {{
    const textarea = document.querySelector('#{elem_id} textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}}
"""

theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}

/* CSS for Pause Tags */
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""


def resolve_seed(seed) -> int:
    """-1 (or blank) picks a random seed; the seed used is reported so a take can be repeated."""
    try:
        resolved = int(seed)
    except (TypeError, ValueError):
        resolved = -1
    return random.randint(0, 2**31 - 1) if resolved < 0 else resolved % 2**32


def output_file_name(text: str, language: str, extension: str) -> Path:
    stem = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    stem = re.sub(r"[\s-]+", "_", stem).strip("_").lower()[:20] or "narration"
    lang = re.sub(r"[^\w]", "", (language or "auto").lower()) or "auto"
    return OUTPUT_DIR / f"{stem}_{lang}_{uuid.uuid4().hex[:8].upper()}.{extension}"


def _status_line(result, seed: int, elapsed_s: float) -> str:
    head = "Done." if result.complete else "⚠️ Stopped early, partial audio saved."
    parts = [f"{head} Seed {seed}", f"{result.duration_s / 60:.1f} min of audio in {elapsed_s / 60:.1f} min"]
    if result.chunk_count > 1:
        parts.append(f"joined {result.chunk_count} chunks")
    status = " · ".join(parts)
    return "\n".join([status, *(f"⚠️ {w}" for w in result.warnings)])


def make_generate_fn(narrator: MossNarrator, clone: bool):
    def generate(text, language, ref_audio, out_format, temperature, top_p, top_k,
                 repetition_penalty, seed, chunk_chars, sentence_pause, paragraph_pause,
                 voice_lock, progress=gr.Progress()):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize.", None
        if clone and not ref_audio:
            return None, "Please upload a reference audio.", None
        used_seed = resolve_seed(seed)
        settings = NarrationSettings(
            max_chunk_weight=int(chunk_chars), sentence_gap_s=float(sentence_pause),
            paragraph_gap_s=float(paragraph_pause), temperature=float(temperature),
            top_p=float(top_p), top_k=int(top_k), repetition_penalty=float(repetition_penalty),
            voice_lock=bool(voice_lock), seed=used_seed,
        )

        def on_progress(done: int, total: int, _chunk: str) -> None:
            progress(done / total, desc=f"Generating chunk {min(done + 1, total)}/{total}")

        started = time.monotonic()
        try:
            result = narrator.narrate(text, ref_audio if clone else None, language, settings, on_progress)
        except Exception as error:  # show every failure in the Status box
            logging.exception("Generation failed")
            return None, f"❌ {error}", None
        path = save_audio(result.audio, result.sample_rate, output_file_name(text, language, out_format.lower()))
        return str(path), _status_line(result, used_seed, time.monotonic() - started), str(path)

    return generate


def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(label=label, choices=list(LANGUAGES), value=value, allow_custom_value=False,
                       interactive=True, info="Auto works for English and Chinese. Pick the language for others.")


def _tag_buttons(textbox: gr.Textbox, elem_id: str) -> None:
    with gr.Row(elem_classes=["tag-container"]):
        for tag in PAUSE_TAGS:
            btn = gr.Button(tag, elem_classes=["tag-btn"])
            btn.click(fn=None, inputs=[btn, textbox], outputs=textbox, js=INSERT_TAG_JS.format(elem_id=elem_id))


def _gen_settings(show_voice_lock: bool):
    with gr.Accordion("Generation Settings (optional)", open=False):
        fmt = gr.Radio(["MP3", "WAV"], value="MP3", label="Output Format")
        temp = gr.Slider(0.5, 2.0, value=DEFAULTS.temperature, step=0.05, label="Temperature",
                         info="Lower = steadier delivery, higher = more expressive. 1.7 is recommended.")
        top_p = gr.Slider(0.1, 1.0, value=DEFAULTS.top_p, step=0.05, label="Top-p")
        top_k = gr.Slider(1, 100, value=DEFAULTS.top_k, step=1, label="Top-k")
        rep = gr.Slider(1.0, 1.5, value=DEFAULTS.repetition_penalty, step=0.05, label="Repetition Penalty")
        seed = gr.Number(value=-1, precision=0, label="Seed",
                         info="-1 picks a new voice every run. Copy the seed from Status to repeat a take exactly.")

        with gr.Accordion("Long text", open=False):
            chunk = gr.Slider(150, 1500, value=DEFAULTS.max_chunk_weight, step=50, label="Max characters per chunk",
                              info="Long text is split on sentence boundaries and joined back together.")
            s_pause = gr.Slider(0.0, 2.0, value=DEFAULTS.sentence_gap_s, step=0.05, label="Pause between sentences (seconds)")
            p_pause = gr.Slider(0.0, 4.0, value=DEFAULTS.paragraph_gap_s, step=0.1, label="Pause between paragraphs (seconds)")
            lock = gr.Checkbox(value=DEFAULTS.voice_lock, visible=show_voice_lock, label="Keep the same voice for the whole script",
                               info="Makes a short warm-up take to pick the voice, then reads every chunk in it. This also stops MOSS-TTS adding intro music.")
    return [fmt, temp, top_p, top_k, rep, seed, chunk, s_pause, p_pause, lock]


def _output_column():
    with gr.Column(scale=1):
        audio = gr.Audio(label="Output Audio", type="filepath")
        status = gr.Textbox(label="Status", lines=2)
        with gr.Accordion("Download files", open=False):
            file = gr.File(label="Generated Audio")
    return audio, status, file


def _tab(narrator: MossNarrator, name: str, clone: bool) -> None:
    elem_id = f"{name}_textbox"
    with gr.TabItem("Voice Clone" if clone else "Auto Voice"):
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="Text to Synthesize", lines=10, elem_id=elem_id,
                                  placeholder="Paste your full narration here. Blank lines between paragraphs become longer pauses.")
                _tag_buttons(text, elem_id)
                lang = _lang_dropdown()
                ref = gr.Audio(label="Reference Audio (5–30 seconds of clean speech)", type="filepath",
                               elem_classes="compact-audio", visible=clone)
                btn = gr.Button("Generate", variant="primary")
                settings = _gen_settings(show_voice_lock=not clone)
            audio, status, file = _output_column()
        btn.click(make_generate_fn(narrator, clone), inputs=[text, lang, ref, *settings],
                  outputs=[audio, status, file])


def build_demo(narrator: MossNarrator) -> gr.Blocks:
    with gr.Blocks(theme=theme, css=css, title="MOSS-TTS 8B Narrator") as demo:
        gr.HTML("""
            <div style="text-align: center; margin: 20px auto; max-width: 800px;">
                <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ MOSS-TTS 8B Narrator</h1>
                <p>Long-form narration with MOSS-TTS-v1.5 (8B) in 31 languages, with Voice Clone and Auto Voice.
                Paste a full script and get one audio file.</p>
            </div>
        """)
        with gr.Tabs():
            _tab(narrator, "vc", clone=True)
            _tab(narrator, "av", clone=False)
    return demo


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_demo(MossNarrator.load())
    demo.queue().launch(share=True, debug=True, allowed_paths=[str(OUTPUT_DIR)])
