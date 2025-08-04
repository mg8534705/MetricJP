# main.py

import argparse
from modules.asr import start_asr_stream
from modules.nmt import Translator
from modules.tts import TTSGenerator
from modules.utils.audio_io import play_audio
from modules.utils.logging import setup_logger

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(description="Real-time Japanese-English Speech Translator")
    parser.add_argument("--source_lang", choices=["ja", "en"], required=True, help="Source language: ja or en")
    args = parser.parse_args()

    logger.info("Starting real-time translation system")

    # Initialize components
    translator = Translator(source_lang=args.source_lang)
    tts = TTSGenerator(target_lang='en' if args.source_lang == 'ja' else 'ja')

    # Start ASR stream
    for recognized_text in start_asr_stream(lang=args.source_lang):
        logger.info(f"Recognized ({args.source_lang}): {recognized_text}")

        # Translate
        translated_text = translator.translate(recognized_text)
        logger.info(f"Translated: {translated_text}")

        # Generate and play audio
        audio_waveform = tts.synthesize(translated_text)
        play_audio(audio_waveform)

if __name__ == "__main__":
    main()
