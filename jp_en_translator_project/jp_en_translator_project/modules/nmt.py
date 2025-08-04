import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class Translator:
    def __init__(self, source_lang="ja"):
        self.source_lang = source_lang
        self.target_lang = "en" if source_lang == "ja" else "ja"
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer.src_lang = "ja_XX" if source_lang == "ja" else "en_XX"
        self.target_lang_code = "en_XX" if source_lang == "ja" else "ja_XX"

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang_code])
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
