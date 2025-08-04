import torch
import numpy as np
from TTS.api import TTS

class TTSGenerator:
    def __init__(self, target_lang="en"):
        self.lang = target_lang
        self.tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=False, gpu=torch.cuda.is_available())

    def synthesize(self, text: str) -> np.ndarray:
        return self.tts.tts(text=text, speaker=0)
