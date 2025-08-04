import whisper
import sounddevice as sd
import numpy as np
import queue

model = whisper.load_model("base")

def start_asr_stream(lang="ja"):
    audio_q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_q.put(indata.copy())

    with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
        buffer = np.zeros((16000 * 5,), dtype=np.float32)
        offset = 0
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                audio_chunk = audio_q.get()
                buffer[offset:offset + len(audio_chunk)] = audio_chunk[:, 0]
                offset += len(audio_chunk)

                if offset >= len(buffer):
                    result = model.transcribe(buffer, language=lang)
                    yield result["text"]
                    offset = 0
        except KeyboardInterrupt:
            print("Stopping ASR.")
