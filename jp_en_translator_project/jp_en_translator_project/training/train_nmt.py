import json
import os
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return Dataset.from_dict({ "translation": pairs })

def preprocess(example, tokenizer, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(example['translation'][src_lang], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(example['translation'][tgt_lang], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

def train():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    data = load_data("jp_en_translator_project/demo_data/prefix_aligned_data.json")
    data = data.map(lambda x: preprocess(x, tokenizer, "ja", "en"))

    args = Seq2SeqTrainingArguments(
        output_dir="nmt_model",
        evaluation_strategy="no",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_total_limit=1,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=[]
    )

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=data, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model("nmt_model")

if __name__ == "__main__":
    train()
