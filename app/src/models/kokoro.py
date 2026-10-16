import os
import numpy as np
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
from datetime import datetime

from tqdm import tqdm

from src.config import Configuration
import warnings

warnings.filterwarnings("ignore")


def kokoro_generate_audio(text, CONFIG: Configuration, print_output: bool = False):
    """
    Generate audio using Kokoro TTS model and configuration.
    
    :param text: Text to be converted to speech
    :type text: str
    :param CONFIG: Configuration object containing TTS settings
    :type CONFIG: Configuration
    
    """
    pipeline = KPipeline(lang_code=CONFIG.kokoro_language, repo_id=CONFIG.kokoro_repo_id)
    generator = pipeline(
        text, 
        voice=CONFIG.kokoro_voice,
        speed=CONFIG.kokoro_play_speed,
        split_pattern=r'\n+'
    )

    all_audio = []
    for i, (gs, ps, audio) in tqdm(enumerate(generator), desc="Generating audio chunks"):
        if print_output:
            print(f" - Chunk {i}: {gs}")
            display(Audio(data=audio, rate=CONFIG.kokoro_rate, autoplay=i==0))
        
        all_audio.append(audio)

    # COMBINE all chunks into one audio stream
    final_audio = np.concatenate(all_audio)

    save_path = os.path.join(
        CONFIG.OUTPUT_PATH, 
        f'kokoro-{CONFIG.kokoro_voice}-{os.path.basename(CONFIG.audio_name)}.wav'
    )
    sf.write(save_path, final_audio, CONFIG.kokoro_rate)

    print(f" - Audio saved at: {save_path}")

    return final_audio, save_path