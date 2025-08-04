import os
import subprocess
import sys

def check_and_download_models():
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    from TTS.api import TTS
    import whisper

    MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    TTS(model_name="tts_models/en/ljspeech/fast_pitch")
    whisper.load_model("base")
    print("✅ All models ready.")

def launch_gui():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "jp_en_translator_project/gui/translator_gui.py"])

if __name__ == "__main__":
    check_and_download_models()
    launch_gui()
