import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

from src.config import Configuration


def qwen3_generate_audio(text, CONFIG: Configuration):
    """
    Generate audio using Qwen3-TTS.

    :param text: Text to be converted to audio
    :param CONFIG: Configuration object
    :type CONFIG: Configuration
    """
    print(" - Loading Qwen3-TTS...")
    model = Qwen3TTSModel.from_pretrained(
        CONFIG.qwen3_pretrained_model_name_or_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    print(" - Generating: text...")
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=CONFIG.qwen3_language, 
        speaker=CONFIG.qwen3_speaker, 
    )
    
    sf.write(CONFIG.save_path, wavs[0], sr)
    print(f" - Saved to {CONFIG.save_path}")
    

    return wavs[0], CONFIG.save_path