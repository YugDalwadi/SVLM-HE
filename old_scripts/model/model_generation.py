# ----------------------------
# 1. Imports
# ----------------------------
import os
import pandas as pd
import numpy as np
import json
from math import log
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

# ----------------------------
# 2. Paths
# ----------------------------
mimic_path = "/raid/deeksha/mimic/mimic-iv-3.1"
note_path = "/raid/deeksha/mimic/mimic_note/note/"

# ----------------------------
# 3. Load structured + note data
# ----------------------------
patients = pd.read_csv(os.path.join(mimic_path, "hosp/patients.csv.gz"))
admissions = pd.read_csv(os.path.join(mimic_path, "hosp/admissions.csv.gz"))
diagnoses = pd.read_csv(os.path.join(mimic_path, "hosp/diagnoses_icd.csv.gz"))
d_icd = pd.read_csv(os.path.join(mimic_path, "hosp/d_icd_diagnoses.csv.gz"))

# HE cohort
he_codes = d_icd[d_icd["long_title"].str.contains("hepatic encephalopathy", case=False, na=False)]
he_adm = diagnoses.merge(he_codes, on=["icd_code", "icd_version"], how="inner")["hadm_id"].unique()
he_adm = set(he_adm)
admissions["label"] = admissions["hadm_id"].apply(lambda x: 1 if x in he_adm else 0)
admissions = admissions.merge(patients[["subject_id", "anchor_age"]], on="subject_id", how="left")
admissions = admissions[admissions["anchor_age"] >= 18]

# Notes
discharge = pd.read_csv(os.path.join(note_path, "discharge.csv.gz"))
radiology = pd.read_csv(os.path.join(note_path, "radiology.csv.gz"))

print(discharge.head())
print(radiology.head())

print(discharge.columns)
print(radiology.columns)

notes = pd.concat([
    discharge[["subject_id", "hadm_id", "text"]].rename(columns={"text": "discharge_note"}),
    radiology[["subject_id", "hadm_id", "text"]].rename(columns={"text": "radiology_note"})
], axis=0)

notes_grouped = notes.groupby(["subject_id", "hadm_id"]).agg(lambda x: " ".join(x.astype(str))).reset_index()
admissions = admissions.merge(notes_grouped, on=["subject_id", "hadm_id"], how="left")

# ----------------------------
# 4. Generate JSONL
# ----------------------------
out_records = []
for _, row in admissions.iterrows():
    label = "HE" if row["label"] == 1 else "non-HE"
    text_note = ""
    if "discharge_note" in row and pd.notna(row["discharge_note"]):
        text_note += f"\nDischarge Summary:\n{row['discharge_note']}"
    if "radiology_note" in row and pd.notna(row["radiology_note"]):
        text_note += f"\nRadiology Report:\n{row['radiology_note']}"

    record = {
        "instruction": "Classify if this admission has hepatic encephalopathy (HE) based on labs and clinical notes.",
        "input": {
            "age": row["anchor_age"],
            "hadm_id": int(row["hadm_id"]),
            "notes": text_note.strip()
        },
        "output": label
    }
    out_records.append(record)

jsonl_path = "/raid/deeksha/mimic/he_notes_classification.jsonl"
with open(jsonl_path, "w") as f:
    for r in out_records:
        f.write(json.dumps(r) + "\n")

print("Saved dataset with notes:", len(out_records))


# ----------------------------
# 5. Load dataset & balance
# ----------------------------
dataset = load_dataset("json", data_files=jsonl_path, split="train")
he_ds = dataset.filter(lambda ex: ex["output"] == "HE")
nonhe_ds = dataset.filter(lambda ex: ex["output"] == "non-HE")
nonhe_ds = nonhe_ds.shuffle(seed=42).select(range(len(he_ds)))  # undersample majority

from datasets import concatenate_datasets

balanced_ds = concatenate_datasets([
    he_ds.shuffle(seed=42),
    nonhe_ds.shuffle(seed=42)
]).shuffle(seed=42)

balanced_jsonl_path = "/raid/deeksha/mimic/he_notes_classification_balanced.jsonl"

# HuggingFace datasets has built-in export
balanced_ds.to_json(balanced_jsonl_path, orient="records", lines=True)

print("Saved balanced dataset:", len(balanced_ds), "examples")

# Train-test split
train_test = balanced_ds.train_test_split(test_size=0.2, seed=42)
train_ds, eval_ds = train_test["train"], train_test["test"]


# ----------------------------
# 6. Tokenize
# ----------------------------
model_name = "/raid/deeksha/mimic/models/xz97/AlpaCare-llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def preprocess(ex):
    text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{json.dumps(ex['input'])}\n\n### Response:\n{ex['output']}"
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = train_ds.map(preprocess, remove_columns=dataset.column_names)
eval_ds = eval_ds.map(preprocess, remove_columns=dataset.column_names)

# ----------------------------
# 7. Data collator
# ----------------------------
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ----------------------------
# 8. Training arguments
# ----------------------------
args = TrainingArguments(
    output_dir="./alpacare-he-notes",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=100,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=False,
    push_to_hub=False
)

# ----------------------------
# 9. Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer
)

trainer.train()

# ----------------------------
# 10. Inference example
# ----------------------------
test_example = eval_ds.shuffle(seed=42)[0]
prompt = f"""### Instruction:
{test_example['instruction']}

### Input:
{json.dumps(test_example['input'])}

### Response:
"""
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=50)
print("Model output:\n", tokenizer.decode(output[0], skip_special_tokens=True))
