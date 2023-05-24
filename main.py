import json

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features
from datasets.features.translation import Translation
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline,
)

device = torch.cuda.current_device() if torch.cuda.is_available() else -1

# available models: "facebook/nllb-200-distilled-600M", "facebook/nllb-200-1.3B", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-3.3B"
checkpoint = "facebook/nllb-200-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, src_lang="eng_Latn", tgt_lang="kor_Hang"
)
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",
    tgt_lang="kor_Hang",
    device=device,
)

# Load "easyword" dataset
with open("dump.json") as f:
    data = json.load(f)

# Try translation
jargons = [entry["english"] for entry in data]
for jargon in jargons:
    target_seq = translator(jargon, max_length=128)
    print(jargon, target_seq[0]["translation_text"])


translations = {entry["english"]: list(entry["translations"].keys()) for entry in data}
processed_data = []
for english, translation in translations.items():
    for words in translation:
        if "," in words:
            words = words.split(",")
        else:
            words = [words]
        processed_data += [(english, korean) for korean in words]


# Preprocess for fine-tuning
df = pd.DataFrame(
    {
        "src": [english for english, _ in processed_data],
        "trg": [korean for _, korean in processed_data],
    }
)
df = df.sample(frac=1, random_state=42)  # shuffle

df_len = len(df)
train_len = int(df_len * 0.6)
valid_len = int(df_len * 0.2)
test_len = df_len - train_len - valid_len

train_df, valid_df, test_df = (
    df.iloc[:train_len, :],
    df.iloc[train_len : train_len + valid_len, :],
    df.iloc[train_len + valid_len :, :],
)


finetune_ds = DatasetDict()
trans_features = Features({"translation": Translation(languages=("en", "ko"))})
finetune_ds["train"], finetune_ds["validation"], finetune_ds["test"] = (
    Dataset.from_pandas(
        pd.DataFrame(
            {
                "translation": [
                    {"en": src, "ko": trg} for src, trg in zip(df.src, df.trg)
                ]
            }
        ),
        features=trans_features,
    )
    for df in (train_df, valid_df, test_df)
)

# Prepare for fine-tuning
source_lang = "en"
target_lang = "ko"
prefix = f"{source_lang}_{target_lang}"
max_input_length = 128
max_target_length = 128


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_input_length, truncation=True
    )
    return model_inputs


tokenized_datasets = finetune_ds.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result = {"bleu": result["score"]}
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# Train
training_args = Seq2SeqTrainingArguments(
    output_dir="easyword_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

trainer.save_model()
trainer.evaluate(eval_dataset=tokenized_datasets["test"])

# Try again with the fine-tuned model
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",
    tgt_lang="kor_Hang",
    device=device,
)
for jargon in jargons:
    target_seq = translator(jargon, max_length=128)
    print(jargon, target_seq[0]["translation_text"])


trainer.push_to_hub()
