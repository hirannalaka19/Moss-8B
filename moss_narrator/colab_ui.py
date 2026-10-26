"""ipywidgets front-end for Google Colab: paste text, click Generate, download one file."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import ipywidgets as widgets
from IPython.display import Audio, display

from .audio_utils import save_audio
from .engine import MossNarrator
from .planning import LANGUAGES, NarrationSettings, expected_seconds
from .text_chunker import chunk_narration

OUTPUT_DIR = Path("/content/narrations")
# Embedding very long audio in the notebook bloats it; above this we only offer the download.
MAX_PREVIEW_S = 20 * 60
_WIDE = widgets.Layout(width="100%")
_STYLE = {"description_width": "160px"}


def _download(path: Path) -> None:
    try:
        from google.colab import files

        files.download(str(path))
    except ImportError:
        print(f"Saved to {path}")


def _build_advanced() -> tuple[widgets.Accordion, dict[str, widgets.Widget]]:
    defaults = NarrationSettings()
    controls = {
        "max_chunk_weight": widgets.IntSlider(defaults.max_chunk_weight, min=150, max=1500, step=50, description="Chunk size (chars)", style=_STYLE, layout=_WIDE),
        "sentence_gap_s": widgets.FloatSlider(defaults.sentence_gap_s, min=0.0, max=2.0, step=0.05, description="Pause between chunks (s)", style=_STYLE, layout=_WIDE),
        "paragraph_gap_s": widgets.FloatSlider(defaults.paragraph_gap_s, min=0.0, max=4.0, step=0.1, description="Pause between paragraphs (s)", style=_STYLE, layout=_WIDE),
        "temperature": widgets.FloatSlider(defaults.temperature, min=0.5, max=2.0, step=0.05, description="Temperature", style=_STYLE, layout=_WIDE),
        "top_p": widgets.FloatSlider(defaults.top_p, min=0.1, max=1.0, step=0.05, description="Top-p", style=_STYLE, layout=_WIDE),
        "top_k": widgets.IntSlider(defaults.top_k, min=1, max=100, description="Top-k", style=_STYLE, layout=_WIDE),
        "repetition_penalty": widgets.FloatSlider(defaults.repetition_penalty, min=1.0, max=1.5, step=0.05, description="Repetition penalty", style=_STYLE, layout=_WIDE),
        "voice_lock": widgets.Checkbox(defaults.voice_lock, description="Keep the same voice across chunks (when no voice sample)", indent=False),
        "seed": widgets.Text("", description="Seed (blank = random)", style=_STYLE),
    }
    accordion = widgets.Accordion(children=[widgets.VBox(list(controls.values()))])
    accordion.set_title(0, "Advanced settings")
    accordion.selected_index = None
    return accordion, controls


def _settings_from(controls: dict[str, widgets.Widget]) -> NarrationSettings:
    seed_text = controls["seed"].value.strip()
    if seed_text and not (seed_text.isdigit() and int(seed_text) < 2**32):
        raise ValueError("Seed must be blank or a whole number from 0 to 4294967295.")
    values = {name: widget.value for name, widget in controls.items() if name != "seed"}
    return NarrationSettings(**values, seed=int(seed_text) if seed_text else None)


def _render(narrator: MossNarrator, ui: dict, out: widgets.Output) -> None:
    text = ui["text"].value
    settings = _settings_from(ui["advanced"])
    chunks = chunk_narration(text, settings.max_chunk_weight)
    if not chunks:
        raise ValueError("Paste some narration text first.")
    reference = ui["reference"].value.strip() or None
    print(f"{len(chunks)} chunks, about {expected_seconds(text) / 60:.1f} min of audio expected.")
    print("Voice: " + (f"cloned from {reference}" if reference else "auto (locked after chunk 1)"))

    started = time.monotonic()

    def on_progress(done: int, total: int, chunk_text: str) -> None:
        ui["progress"].max = total
        ui["progress"].value = done
        elapsed = time.monotonic() - started
        ui["status"].value = f"Chunk {min(done + 1, total)}/{total} | {elapsed / 60:.1f} min elapsed"
        if chunk_text:
            print(f"\n[{done + 1}/{total}] {chunk_text[:90]}{'…' if len(chunk_text) > 90 else ''}")

    result = narrator.narrate(text, reference, ui["language"].value, settings, on_progress)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = save_audio(result.audio, result.sample_rate, OUTPUT_DIR / f"narration-{stamp}.{ui['format'].value}")
    for warning in result.warnings:
        print("⚠️", warning)
    minutes = (time.monotonic() - started) / 60
    if result.complete:
        ui["status"].value = f"✅ Done: {result.duration_s / 60:.1f} min of audio in {minutes:.1f} min"
    else:
        ui["status"].value = f"⚠️ Stopped early: saved a partial file with {result.duration_s / 60:.1f} min of audio (see warnings below)"
    print(f"\nSaved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("If the download doesn't start, open the 📁 Files panel on the left → narrations → right-click → Download.")
    if result.duration_s <= MAX_PREVIEW_S:
        display(Audio(str(path)))
    _download(path)


def launch(narrator: MossNarrator, reference_path: str | None = None) -> None:
    """Render the narration form in the current notebook cell."""
    accordion, advanced = _build_advanced()
    ui = {
        "text": widgets.Textarea(placeholder="Paste your full narration here…", layout=widgets.Layout(width="100%", height="320px")),
        "language": widgets.Dropdown(options=LANGUAGES, value="Auto", description="Language", style=_STYLE),
        "reference": widgets.Text(reference_path or "", placeholder="optional: /content/my_voice.wav", description="Voice sample file", style=_STYLE, layout=_WIDE),
        "format": widgets.ToggleButtons(options=[("MP3", "mp3"), ("WAV", "wav")], value="mp3", description="Output"),
        "progress": widgets.IntProgress(value=0, min=0, max=1, layout=_WIDE),
        "status": widgets.HTML(""),
        "advanced": advanced,
    }
    button = widgets.Button(description="Generate narration", button_style="success", icon="microphone", layout=widgets.Layout(width="240px", height="44px"))
    out = widgets.Output()

    def on_click(_):
        button.disabled = True
        out.clear_output()
        with out:
            try:
                _render(narrator, ui, out)
            except Exception as error:  # surface every failure in the notebook, never silently
                ui["status"].value = f"❌ {error}"
                traceback.print_exc()
            finally:
                button.disabled = False

    button.on_click(on_click)
    display(widgets.VBox([
        ui["text"], ui["language"], ui["reference"], ui["format"], accordion,
        button, ui["progress"], ui["status"], out,
    ]))
