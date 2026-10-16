"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse

from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator, print_color

from src.config import Configuration
from src.models import whisper_transcribe, kokoro_generate_audio, ask_rag, qwen3_generate_audio

def cmd_read_audio(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    
    print_separator("FULL PIPELINE STARTED", sep_type="START")
    # ======================================================================================
    #                                       WHISPER
    # ======================================================================================
    print_separator("WHISPER", sep_type="LONG")
    transcripcion = whisper_transcribe(CONFIG)
    if CONFIG.verbose:
        # transcripcion="What is TARDIS in Deutsche Telekom?"
        print_separator("Wihsper trascription")
        print_color(transcripcion, color="green")
        
    # ======================================================================================
    #                                       RAG
    # ======================================================================================
    print_separator("RAG", sep_type="LONG")
    rag_response = ask_rag(transcripcion, CONFIG)

    if CONFIG.verbose:
        print_separator("RAG response")
        print_color(rag_response, color="green")

    # ======================================================================================
    #                                       TTS
    # ======================================================================================
    if CONFIG.tts_model_name == "qwen3":
        print_separator("QWEN3 TTS", sep_type="LONG")
        final_audio, save_path = qwen3_generate_audio(rag_response, CONFIG)
    elif CONFIG.tts_model_name == "kokoro":
        print_separator("KOKORO", sep_type="LONG")
        final_audio, save_path = kokoro_generate_audio(rag_response, CONFIG)
    else:
        raise ValueError(f"Unknown TTS model name: {CONFIG.tts_model_name}")
    
    print_separator("FULL PIPELINE ENDED", sep_type="START")
    
    return final_audio, save_path


def cmd_whisper(args: argparse.Namespace):
    """Call Whisper function with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("WHISPER STARTED", sep_type="START")
    whisper_transcribe(CONFIG)
    print_separator("WHISPER ENDED", sep_type="START")

def cmd_rag(args: argparse.Namespace):
    """Call RAG function with the given args."""
    print_separator("RAG STARTED", sep_type="START")
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    ask_rag(args.query, CONFIG)
    print_separator("RAG ENDED", sep_type="START")


def cmd_kokoro(args: argparse.Namespace):
    """Call Kokoro function with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("KOKORO STARTED", sep_type="START")
    kokoro_generate_audio(args.text, CONFIG)
    print_separator("KOKORO ENDED", sep_type="START")

def cmd_qwen3(args: argparse.Namespace):
    """Call Qwen3 TTS function with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("QWEN3 TTS STARTED", sep_type="START")
    qwen3_generate_audio(args.text, CONFIG)
    print_separator("QWEN3 TTS ENDED", sep_type="START")
    
# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="voice-assistant", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)


    # ======================================================================================
    #                                       FULL PIPELINE
    # ======================================================================================
    p_full = subparsers.add_parser("full", help="Test script for full pipeline")
    p_full.add_argument(
        "-a", "--audio_name", type=str, required=True, help="Path to the audio file"
    )
    p_full.add_argument(
        "-v", "--verbose", action="store_false", help="Disable verbose output", required=False
    )
    p_full.set_defaults(func=cmd_read_audio)

    # ======================================================================================
    #                                       WHISPER
    # ======================================================================================
    p_whisper = subparsers.add_parser("whisper", help="Test script for Whisper ASR")
    p_whisper.add_argument(
        "-a", "--audio_name", type=str, required=True, help="Path to the audio file"
    )
    p_whisper.set_defaults(func=cmd_whisper)

    # ======================================================================================
    #                                       RAG
    # ======================================================================================
    p_rag = subparsers.add_parser("rag", help="Test script with RAG")
    p_rag.add_argument(
        "-q", "--query", type=str, required=True, help="Query to ask the RAG system"
    )
    p_rag.set_defaults(func=cmd_rag)

    # ======================================================================================
    #                                       KOKORO
    # ======================================================================================
    p_kokoro = subparsers.add_parser("kokoro", help="Test script with Kokoro")
    p_kokoro.add_argument(
        "-t", "--text", type=str, required=True, help="Query to ask the Kokoro system"
    )
    p_kokoro.set_defaults(func=cmd_kokoro, tts_model_name="kokoro")

    # ======================================================================================
    #                                       QWEN3 TTS
    # ======================================================================================
    p_qwen3 = subparsers.add_parser("qwen3", help="Test script with Qwen3 TTS")
    p_qwen3.add_argument(
        "-t", "--text", type=str, required=True, help="Query to ask the Qwen3 TTS system"
    )
    p_qwen3.set_defaults(func=cmd_qwen3, tts_model_name="qwen3")


    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)