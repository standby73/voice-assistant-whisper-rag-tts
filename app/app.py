"""Streamlit Web UI for the audio pipeline.

Run from ./app directory:
    streamlit run app_streamlit.py
"""

import os
import datetime
from pathlib import Path
import streamlit as st
import dotenv
from audio_recorder_streamlit import audio_recorder

from main import cmd_read_audio

from maikol_utils.file_utils import make_dirs

# ======================================================================================
#                                       PATHS
# ======================================================================================
AUDIO_INPUT_DIR = os.path.abspath(os.path.join("..", "data", "audios"))
OUTPUT_DIR = os.path.abspath(os.path.join("..", "output"))

make_dirs([AUDIO_INPUT_DIR, OUTPUT_DIR])


# ======================================================================================
#                                       PROCESSING
# ======================================================================================
def process_audio(audio_filepath: str, tts_model: str = "kokoro") -> tuple[str | None, str]:
    """Process the audio file through the pipeline.
    
    Pipeline: Whisper (transcription) → RAG (response) → TTS (audio generation)
    
    Args:
        audio_filepath: Path to the audio file
        tts_model: TTS model to use ("kokoro" or "qwen3")
        
    Returns:
        Tuple: (output_audio_path, status_message)
    """
    if not audio_filepath or not os.path.exists(audio_filepath):
        return None, f"❌ Audio file not found at: {audio_filepath}"
    
    # Verify file size is reasonable
    file_size = os.path.getsize(audio_filepath)
    if file_size == 0:
        return None, "❌ Audio file is empty"
    
    try:
        # Create a minimal args namespace-like object with required configuration
        class Args:
            def __init__(self, audio_path: str, tts_model_name: str):
                self.audio_name = audio_path
                self.verbose = True
                self.seed = 42
                self.tts_model_name = tts_model_name
        
        args = Args(audio_filepath, tts_model)
        
        # Run the full pipeline
        final_audio, save_path = cmd_read_audio(args)
        
        # Return the save path for Streamlit to play
        if save_path and os.path.exists(save_path):
            return save_path, "✅ Processing complete!"
        else:
            return final_audio, "✅ Processing complete!"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


def save_audio_bytes(audio_bytes: bytes, prefix: str = "recording") -> str:
    """Save audio bytes to a file in the audio input directory.
    
    Args:
        audio_bytes: Raw audio bytes
        prefix: Filename prefix
        
    Returns:
        Path to the saved audio file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.wav"
    target_path = os.path.join(AUDIO_INPUT_DIR, filename)
    
    with open(target_path, "wb") as f:
        f.write(audio_bytes)
    
    return target_path


# ======================================================================================
#                                       UI
# ======================================================================================
def main():
    """Main Streamlit application."""
    dotenv.load_dotenv()
    
    # ========================= Page configuration ========================= 
    st.set_page_config(
        page_title="Audio Processing Pipeline",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ Audio Processing Pipeline")
    st.markdown("""
    **Whisper** (transcription) → **RAG** (response) → **Kokoro** (audio generation)
    """)
    st.divider()
    
    # ========================= Initialize session state ========================= 
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "output_audio_path" not in st.session_state:
        st.session_state.output_audio_path = None
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "tts_model" not in st.session_state:
        st.session_state.tts_model = "kokoro"
    
    # ==========================================================================
    #                              UI LAYOUT
    # ==========================================================================
    col_input, col_output = st.columns(2)
    # ========================= INPUT COLUMN =========================
    with col_input:
        st.subheader("📥 Input Audio")
        
        # Tab selection for input method
        tab_record, tab_upload = st.tabs(["🎤 Record Audio", "📁 Upload File"])
        
        # ========================= RECORD TAB =========================
        with tab_record:
            st.markdown("Click the microphone to start/stop recording:")
            
            # Audio recorder widget
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="3x",
                pause_threshold=3.0,  # seconds of silence before auto-stop
                sample_rate=16000
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                # Save recorded audio
                if st.button("✅ Use This Recording", key="use_recording", type="primary"):
                    audio_path = save_audio_bytes(audio_bytes, prefix="recording")
                    st.session_state.audio_path = audio_path
                    st.session_state.output_audio_path = None
                    st.session_state.status_message = f"📁 Audio saved: `{os.path.basename(audio_path)}`"
                    st.rerun()
        
        # ========================= UPLOAD TAB =========================
        with tab_upload:
            st.markdown("Upload an audio file (WAV, MP3, OGG, FLAC):")
            
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "ogg", "flac", "m4a"],
                key="audio_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
                
                if st.button("✅ Use This File", key="use_upload", type="primary"):
                    # Save uploaded file
                    audio_bytes = uploaded_file.read()
                    # Get extension from uploaded file
                    ext = Path(uploaded_file.name).suffix or ".wav"
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"upload_{timestamp}{ext}"
                    target_path = os.path.join(AUDIO_INPUT_DIR, filename)
                    
                    with open(target_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    st.session_state.audio_path = target_path
                    st.session_state.output_audio_path = None
                    st.session_state.status_message = f"📁 Audio saved: `{filename}`"
                    st.rerun()
        
        st.divider()
        
        # ========================= SELECTED AUDIO DISPLAY =========================
        if st.session_state.audio_path:
            st.success(f"**Selected audio:** `{os.path.basename(st.session_state.audio_path)}`")
            
            # Play the selected audio
            if os.path.exists(st.session_state.audio_path):
                st.audio(st.session_state.audio_path)
            
            # Clear button
            if st.button("🗑️ Clear Selection", key="clear_audio"):
                st.session_state.audio_path = None
                st.session_state.output_audio_path = None
                st.session_state.status_message = ""
                st.rerun()
        
        st.divider()
        
        # ========================= TTS MODEL SELECTION =========================
        st.markdown("**🎵 Select TTS Model (Kokoro for fast answer, Qwen3 for better quality answer):**")
        st.session_state.tts_model = st.selectbox(
            "TTS Model",
            options=["kokoro", "qwen3"],
            index=0 if st.session_state.tts_model == "kokoro" else 1,
            help="Choose the Text-to-Speech model for audio generation",
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # ========================= PROCESS BUTTON =========================
        process_disabled = st.session_state.audio_path is None
        
        if st.button(
            "🚀 Process Audio",
            key="process_btn",
            type="primary",
            disabled=process_disabled,
            use_container_width=True
        ):
            st.session_state.processing = True
            st.rerun()
    
    # ==========================================================================
    #                              OUTPUT COLUMN
    # ==========================================================================
    with col_output:
        st.subheader("📤 Output Audio")
        
        # Status message
        if st.session_state.status_message:
            if "✅" in st.session_state.status_message:
                st.success(st.session_state.status_message)
            elif "❌" in st.session_state.status_message:
                st.error(st.session_state.status_message)
            else:
                st.info(st.session_state.status_message)
        
        # Processing logic (runs after rerun from button click)
        if st.session_state.processing and st.session_state.audio_path:
            with st.spinner("🔄 Processing audio through pipeline..."):
                st.info("**Step 1/3:** Whisper transcription...")
                st.info("**Step 2/3:** RAG response generation...")
                tts_name = "Kokoro" if st.session_state.tts_model == "kokoro" else "Qwen3"
                st.info(f"**Step 3/3:** {tts_name} audio synthesis...")
                
                output_path, status = process_audio(st.session_state.audio_path, st.session_state.tts_model)
                
                st.session_state.output_audio_path = output_path
                st.session_state.status_message = status
                st.session_state.processing = False
                st.rerun()
        
        # Display output audio
        if st.session_state.output_audio_path and os.path.exists(st.session_state.output_audio_path):
            st.audio(st.session_state.output_audio_path)
            
            # Download button
            with open(st.session_state.output_audio_path, "rb") as f:
                audio_data = f.read()
            
            st.download_button(
                label="⬇️ Download Output Audio",
                data=audio_data,
                file_name=os.path.basename(st.session_state.output_audio_path),
                mime="audio/wav",
                use_container_width=True
            )
        elif not st.session_state.processing:
            st.info("🎧 Output audio will appear here after processing.")
    
    # ==========================================================================
    #                              FOOTER / DEBUG INFO
    # ==========================================================================
    st.divider()
    
    with st.expander("ℹ️ Debug Information"):
        st.markdown(f"""
        **Working Directory:** `{os.getcwd()}`  
        **Audio Input Directory:** `{AUDIO_INPUT_DIR}`  
        **Output Directory:** `{OUTPUT_DIR}`  
        **Selected Audio:** `{st.session_state.audio_path or 'None'}`  
        **Output Audio:** `{st.session_state.output_audio_path or 'None'}`  
        **TTS Model:** `{st.session_state.tts_model}`
        """)


if __name__ == "__main__":
    main()
