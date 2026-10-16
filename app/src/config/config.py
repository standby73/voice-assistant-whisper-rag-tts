"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass
from typing import Literal

@dataclass 
class Configuration:
    """Configuration class for the project."""

    DATA_PATH: str = os.path.join("..", "data")
    OUTPUT_PATH: str = os.path.join("..", "output")

    audio_path: str = os.path.join(DATA_PATH, "audios")
    pdf_path: str = os.path.join(DATA_PATH, "pdfs")
    db_path: str = os.path.join(DATA_PATH, "db_data")

    audio_name: str = 'sample.wav'  
    save_path: str = os.path.join(OUTPUT_PATH, 'audios')

    whisper_version: str = "turbo"

    rag_model_name: str = "llama3.1"
    rag_embedding_model: str = "nomic-embed-text"
    
    tts_model_name: Literal["kokoro", "qwen3"] = "kokoro"

    kokoro_language: str = "es"
    kokoro_voice: str = "ef_dora"
    kokoro_repo_id: str = 'hexgrad/Kokoro-82M'    
    kokoro_rate: int = 24000
    kokoro_play_speed: float = 1.0

    qwen3_language: str = "Auto"
    qwen3_speaker: str = "Aiden"
    qwen3_pretrained_model_name_or_path : str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    verbose: bool = True

    def __post_init__(self):
        self.audio_name = os.path.join(self.audio_path, self.audio_name)
        
        self.save_path = os.path.join(
            self.OUTPUT_PATH, 
            f'{self.tts_model_name}-{self.kokoro_voice if self.tts_model_name == "kokoro" else self.qwen3_speaker}-{os.path.basename(self.audio_name)}.wav'
        )