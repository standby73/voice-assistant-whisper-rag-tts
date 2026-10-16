import whisper
from src.config import Configuration

def whisper_transcribe(CONFIG: Configuration) -> str:
    """Función para llamar a whisper y transcribir el audio.

    Args:
        path (str): Ruta al audio.
        whisper_model (str, optional): Variante de whisper a utilizar. Defaults to "medium".

    Returns:
        str: Transcripción de whisper.
    """
    model = whisper.load_model(CONFIG.whisper_version)

    result = model.transcribe(CONFIG.audio_name)

    return result["text"]