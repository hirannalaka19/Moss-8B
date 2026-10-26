# 🎙️ MOSS-TTS-v1.5 (8B) Narrator for Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hirannalaka19/Moss-8B/blob/main/MOSS_TTS_Narrator.ipynb)

Paste a full narration script, click a button, and download **one** audio file (MP3 or WAV).
It runs [OpenMOSS **MOSS-TTS-v1.5**](https://github.com/OpenMOSS/MOSS-TTS), the 8B model (`OpenMOSS-Team/MOSS-TTS-v1.5`), on a Colab A100.

## 🚀 How to use

1. Click the **Open In Colab** badge above.
2. Set the runtime to an **A100 GPU**: `Runtime → Change runtime type → A100 GPU`. This needs Colab Pro or Pay As You Go; T4 and L4 are too small for the 8B model.
3. **One-time:** add your Hugging Face token to Colab **Secrets**. Click the 🔑 icon in the left sidebar → *Add new secret* → Name: `HF_TOKEN`, Value: your token (create a free *Read* token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) → enable *Notebook access*. Without it, downloads are throttled.
4. Run the **Install MOSS-TTS** cell.
5. Run the **Run Gradio APP** cell. The first run downloads about 24 GB of weights (5 to 10 minutes). Then open the public `*.gradio.live` link.

## 🖥️ The app

Two tabs, both built for long scripts:

* **Voice Clone**: paste your narration, upload 5 to 30 seconds of speech (or record with the mic), and click **Generate**. Background sound in the sample is removed automatically; listen to the **Cleaned reference** player to check it.
* **Auto Voice**: paste your narration and click **Generate**. The model picks a voice, and it stays the same for the whole script.

Click the `[pause 0.5s]` … `[pause 3.0s]` buttons to insert a pause at the cursor. The result appears under **Output Audio**, and the file is under **Download files**. The **Status** box shows the seed used; enter it under **Generation Settings → Seed** to repeat a take exactly.

## How it works

| Step | What happens |
|---|---|
| Chunking | The text is split by paragraph, then sentences are grouped into chunks of about 500 characters (roughly 35 s of speech). Very long sentences are split at commas, then at words. Chinese, Japanese and Korean text is supported. |
| Voice consistency | If you upload a voice sample, it is encoded once and used for every chunk. Without a sample, chunk 1 is generated in the model's own voice, and that voice is reused for every later chunk so the whole narration sounds like one speaker. |
| Quality checks | If a chunk's length doesn't match its text (cut short, or running on), it is regenerated, up to 2 retries. |
| Stitching | Silence at each chunk's edges is trimmed, loudness is evened out, and the chunks are joined with a 0.3 s pause between sentences and 0.8 s between paragraphs. Where a long sentence had to be split, the pause is only 0.08 s. |
| Reference cleanup | MOSS-TTS copies the room and background of the voice sample along with the voice. With **Remove background from reference audio** on (the default), [Demucs](https://github.com/adefossez/demucs) (`htdemucs_ft`) isolates the voice before cloning, and the **Cleaned reference** player lets you hear exactly what the model will copy. In tests on speech mixed with backgrounds at 5 dB SNR, it reached 17.5 dB with music, 19.3 dB with hum and hiss, and 10.3 dB with crowd or traffic noise. Clean samples pass through unchanged. |
| Denoise | MOSS-TTS leaves a faint low rumble and a voice-pitched buzz in pauses. With **Denoise** on (the default), a 60 Hz high-pass and spectral noise reduction ([noisereduce](https://github.com/timsainb/noisereduce)) remove it, using a noise sample taken from the narration's own pauses. On a 12-minute test narration, pause noise fell from −61 to −75 dBFS while the voice changed by about 1 dB. |
| Crash safety | If something fails partway through (for example, the GPU runs out of memory), the chunks already finished are still saved as a partial file, so a long run is never lost entirely. |

Settings use the upstream recommended decoding values for the 8B model (temperature 1.7, top-p 0.8, top-k 25). You can change them under **Generation Settings**, and chunk size and pauses under **Generation Settings → Long text**.

### Tips

- MOSS-TTS-v1.5 understands inline pauses such as `[pause 1.5s]`.
- For languages other than Chinese or English, pick the language from the dropdown. Upstream recommends a language tag for v1.5.
- Blank lines between paragraphs become longer pauses.
- Speed on an A100 is very roughly real-time or a little slower, so a 10-minute narration takes about 10 to 20 minutes. The progress bar shows each chunk as it finishes.

## Requirements

| GPU | Works? |
|---|---|
| A100 40/80 GB, H100 | ✅ |
| L4 (24 GB), T4 (15 GB) | ❌ The 8B model in bf16 (~17 GB) plus its audio codec (~7 GB) needs about 25 GB. The notebook stops with a message telling you to switch to an A100. |

## Repository layout

```
MOSS_TTS_Narrator.ipynb     Colab notebook: Install + Run Gradio APP
app.py                      Gradio web UI (Voice Clone / Auto Voice tabs)
colab.txt                   extra packages installed on Colab
moss_narrator/
  text_chunker.py           paragraph, sentence and clause chunking
  planning.py               settings, token budgets, duration checks, GPU check
  audio_utils.py            ffmpeg load, trim, loudness, stitch, MP3/WAV export
  engine.py                 MOSS-TTS model loading, retries, voice lock, narration
  voice_cleaner.py          Demucs background removal for voice-clone references
tests/                      pytest suite (runs on CPU, no model needed)
```

## 💻 Run locally (needs a GPU with 30+ GB)

```bash
git clone https://github.com/hirannalaka19/Moss-8B.git
cd Moss-8B
pip install -r requirements.txt
python app.py
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest --cov=moss_narrator
```

## Credits and license

- Model: [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), Apache-2.0. Check the model license before commercial use, and don't clone anyone's voice without their consent.
- This wrapper: MIT (see `LICENSE`).
