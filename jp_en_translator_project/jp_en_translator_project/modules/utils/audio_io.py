import sounddevice as sd

def play_audio(audio, sample_rate=22050):
    sd.play(audio, sample_rate)
    sd.wait()
