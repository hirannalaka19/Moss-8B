"""Long-form narration with MOSS-TTS-v1.5 (8B) on Google Colab."""

from .planning import MODEL_ID, NarrationSettings
from .text_chunker import Chunk, chunk_narration

__all__ = ["MODEL_ID", "NarrationSettings", "Chunk", "chunk_narration"]
