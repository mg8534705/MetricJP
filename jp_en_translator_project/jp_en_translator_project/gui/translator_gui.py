import streamlit as st
from modules.nmt import Translator
from modules.tts import TTSGenerator
from modules.utils.audio_io import play_audio

st.set_page_config(page_title="Japanese-English Speech Translator", layout="centered")

st.title("🈳 Japanese-English Real-Time Translator")

source_lang = st.selectbox("Select source language", ["ja", "en"], index=0)
text_input = st.text_area("Enter text to translate:")

if st.button("Translate"):
    translator = Translator(source_lang=source_lang)
    tts = TTSGenerator(target_lang='en' if source_lang == 'ja' else 'ja')
    
    translated = translator.translate(text_input)
    st.success(f"**Translated Text:** {translated}")

    audio = tts.synthesize(translated)
    play_audio(audio)
    st.audio(audio, format="audio/wav")
