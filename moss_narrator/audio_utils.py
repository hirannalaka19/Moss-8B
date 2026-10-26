"""Audio helpers: load any format, clean up chunk edges, stitch, and export."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FRAME_MS = 20
EDGE_FADE_MS = 8
PEAK_CEILING = 0.98


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed. On Colab it is preinstalled; locally run `apt install ffmpeg`.")
    return ffmpeg


def load_audio_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode any audio/video file to mono float32 at `sample_rate` using ffmpeg."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    command = [
        _require_ffmpeg(), "-nostdin", "-v", "error", "-i", str(source),
        "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read {source.name}: {result.stderr.decode(errors='replace').strip()}")
    wav = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        raise ValueError(f"{source.name} contains no audio.")
    return wav


def _frame_db(wav: np.ndarray, frame: int) -> np.ndarray:
    count = len(wav) // frame
    frames = wav[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def trim_silence(wav: np.ndarray, sample_rate: int, threshold_db: float = -45.0, keep_ms: int = 80) -> np.ndarray:
    """Remove leading/trailing silence quieter than `threshold_db` below the peak frame."""
    frame = max(1, sample_rate * FRAME_MS // 1000)
    if len(wav) < frame * 2:
        return wav.copy()
    levels = _frame_db(wav, frame)
    voiced = np.nonzero(levels > levels.max() + threshold_db)[0]
    if voiced.size == 0:
        return wav.copy()
    keep = sample_rate * keep_ms // 1000
    start = max(0, voiced[0] * frame - keep)
    end = min(len(wav), (voiced[-1] + 1) * frame + keep)
    return wav[start:end].copy()


def normalize_loudness(wav: np.ndarray, target_dbfs: float = -20.0, max_gain_db: float = 20.0) -> np.ndarray:
    """Scale to a target RMS level. Peaks are left to `soft_limit`, applied once to the whole
    narration, so every chunk gets the same treatment and volume never jumps between chunks."""
    if wav.size == 0:
        return wav.copy()
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < 1e-8:
        return wav.copy()
    gain = min(10.0 ** (target_dbfs / 20.0) / rms, 10.0 ** (max_gain_db / 20.0))
    return (wav * gain).astype(np.float32)


def soft_limit(wav: np.ndarray, ceiling: float = PEAK_CEILING, knee: float = 0.8) -> np.ndarray:
    """Leave samples below `knee` untouched and smoothly compress louder ones under `ceiling`."""
    magnitude = np.abs(wav)
    headroom = ceiling - knee
    compressed = knee + headroom * np.tanh((magnitude - knee) / headroom)
    return np.where(magnitude > knee, np.sign(wav) * compressed, wav).astype(np.float32)


def cut_at_quiet(wav: np.ndarray, sample_rate: int, max_s: float, search_s: float = 2.0) -> np.ndarray:
    """Shorten to at most `max_s`, cutting at the quietest frame of the final `search_s` so words stay whole."""
    limit = int(max_s * sample_rate)
    if len(wav) <= limit:
        return wav.copy()
    frame = max(1, sample_rate * FRAME_MS // 1000)
    window_start = max(0, limit - int(search_s * sample_rate))
    levels = _frame_db(wav[window_start:limit], frame)
    if levels.size == 0:
        return wav[:limit].copy()
    return wav[: window_start + int(np.argmin(levels)) * frame + frame // 2].copy()


def apply_edge_fades(wav: np.ndarray, sample_rate: int, fade_ms: int = EDGE_FADE_MS) -> np.ndarray:
    """Short fade in/out so joins between chunks never click."""
    length = min(len(wav) // 2, sample_rate * fade_ms // 1000)
    out = wav.astype(np.float32, copy=True)
    if length > 0:
        ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
        out[:length] *= ramp
        out[-length:] *= ramp[::-1]
    return out


def stitch(segments: list[np.ndarray], gaps_s: list[float], sample_rate: int) -> np.ndarray:
    """Concatenate segments, inserting gaps_s[i] seconds of silence after segment i."""
    if not segments:
        raise ValueError("No audio segments to stitch.")
    if len(gaps_s) != len(segments) - 1:
        raise ValueError("gaps_s must have exactly one entry between each pair of segments.")
    parts: list[np.ndarray] = [segments[0]]
    for gap, segment in zip(gaps_s, segments[1:]):
        parts.append(np.zeros(int(round(gap * sample_rate)), dtype=np.float32))
        parts.append(segment)
    return np.concatenate(parts).astype(np.float32)


def save_audio(wav: np.ndarray, sample_rate: int, path: str | Path, mp3_bitrate: str = "192k") -> Path:
    """Write .wav (16-bit PCM) or .mp3 depending on the file extension."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        raise ValueError("Output must end in .wav or .mp3")
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2")
    if suffix == ".wav":
        _write_wav(pcm, sample_rate, target)
        return target
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = Path(tmp) / "narration.wav"
        _write_wav(pcm, sample_rate, tmp_wav)
        command = [_require_ffmpeg(), "-nostdin", "-v", "error", "-y", "-i", str(tmp_wav), "-b:a", mp3_bitrate, str(target)]
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"MP3 export failed: {result.stderr.decode(errors='replace').strip()}")
    return target


def _write_wav(pcm: np.ndarray, sample_rate: int, path: Path) -> None:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
