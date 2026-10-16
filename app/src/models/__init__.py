"""Models.

Functions to manage, create, train / test models.
"""
from .whisper import whisper_transcribe
from .kokoro import kokoro_generate_audio
from .rag import ask_rag
from .qwen3 import qwen3_generate_audio