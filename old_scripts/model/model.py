# ----------------------------
# 1. Imports
# ----------------------------
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from math import log

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ----------------------------
# 2. Load tables from MIMIC-IV
# ----------------------------
out_path = "/raid/deeksha/mimic/he_classification.jsonl"

if not os.path.exists(out_path):
    mimic_path = "/raid/deeksha/mimic/mimic-iv-3.1"
    patients = pd.read_csv(os.path.join(mimic_path, "hosp/patients.csv.gz"))
    admissions = pd.read_csv(os.path.join(mimic_path, "hosp/admissions.csv.gz"))
    diagnoses = pd.read_csv(os.path.join(mimic_path, "hosp/diagnoses_icd.csv.gz"))
    d_icd = pd.read_csv(os.path.join(mimic_path, "hosp/d_icd_diagnoses.csv.gz"))
    labevents = pd.read_csv(os.path.join(mimic_path, "hosp/labevents.csv.gz"))

    # ----------------------------
    # 3. Build HE cohort
    # ----------------------------
    # find all HE diagnosis codes
    he_codes = d_icd[d_icd["long_title"].str.contains("hepatic encephalopathy", case=False, na=False)]
    print("HE codes found:", he_codes)

    # admissions with HE
    he_adm = diagnoses.merge(he_codes, on=["icd_code", "icd_version"], how="inner")["hadm_id"].unique()
    he_adm = set(he_adm)

    # label dataset
    admissions["label"] = admissions["hadm_id"].apply(lambda x: 1 if x in he_adm else 0)

    # keep only adult patients
    admissions = admissions.merge(patients[["subject_id", "anchor_age"]], on="subject_id", how="left")
    admissions = admissions[admissions["anchor_age"] >= 18]

    print("HE cases:", admissions["label"].sum(), "Non-HE:", len(admissions) - admissions["label"].sum())


    # ----------------------------
    # 4. Feature assembly (labs snapshot)
    # ----------------------------
    labs_of_interest = {
        "BILIRUBIN TOTAL": "bilirubin",
        "INR(PT)": "inr",
        "ALBUMIN": "albumin",
        "SODIUM": "sodium",
        "CREATININE": "creatinine",
        "AMMONIA": "ammonia"
    }

    items = pd.read_csv(os.path.join(mimic_path, "hosp/d_labitems.csv.gz"))
    items_lookup = items[["itemid", "label"]]

    # filter labevents to selected labs
    lab_sub = labevents.merge(items_lookup, on="itemid", how="left")
    lab_sub = lab_sub[lab_sub["label"].isin(labs_of_interest.keys())]

    # take median per admission
    lab_features = lab_sub.groupby(["hadm_id", "label"])["valuenum"].median().unstack()
    lab_features = lab_features.rename(columns=labs_of_interest).reset_index()

    # merge into admissions
    df = admissions.merge(lab_features, on="hadm_id", how="left")

    # ----------------------------
    # 5. Compute MELD-Na
    # ----------------------------
    def safe_log(x):
        return log(max(x, 1.0))

    def compute_meld_na(row):
        try:
            bilirubin = row.get("bilirubin", np.nan)
            inr = row.get("inr", np.nan)
            creatinine = row.get("creatinine", np.nan)
            sodium = row.get("sodium", np.nan)

            if np.isnan(bilirubin) or np.isnan(inr) or np.isnan(creatinine) or np.isnan(sodium):
                return np.nan

            meld = 3.78*safe_log(bilirubin) + 11.2*safe_log(inr) + 9.57*safe_log(creatinine) + 6.43
            sodium_c = min(max(sodium, 125), 137)
            meld_na = meld + 1.59 * (135 - sodium_c)
            return round(meld_na, 1)
        except:
            return np.nan

    df["meld_na"] = df.apply(compute_meld_na, axis=1)

    # ----------------------------
    # 6. Prepare JSONL for AlpaCare SFT
    # ----------------------------
    out_records = []
    for _, row in df.iterrows():
        features = {
            "age": row["anchor_age"],
            "bilirubin": row.get("bilirubin", None),
            "inr": row.get("inr", None),
            "albumin": row.get("albumin", None),
            "sodium": row.get("sodium", None),
            "creatinine": row.get("creatinine", None),
            "ammonia": row.get("ammonia", None),
            "meld_na": row.get("meld_na", None)
        }
        label = "HE" if row["label"] == 1 else "non-HE"

        record = {
            "instruction": "Classify if this admission has hepatic encephalopathy (HE).",
            "input": features,
            "output": label
        }
        out_records.append(record)

    # save to jsonl
    with open("/raid/deeksha/mimic/he_classification.jsonl", "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    print("Saved dataset:", len(out_records), "examples")

# ----------------------------
# 7. Fine-tuning AlpaCare
# ----------------------------
model_name = "/raid/deeksha/mimic/models/xz97/AlpaCare-llama2-7b"   # <-- replace with your model checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# load dataset
from datasets import load_dataset
from datasets import Dataset

dataset = load_dataset("json", data_files="/raid/deeksha/mimic/he_classification.jsonl", split="train")

he_ds = dataset.filter(lambda ex: ex["output"] == "HE")
nonhe_ds = dataset.filter(lambda ex: ex["output"] == "non-HE")

# Undersample majority (non-HE)
nonhe_ds = nonhe_ds.shuffle(seed=42).select(range(len(he_ds)))

# Merge and shuffle
from datasets import concatenate_datasets

balanced_ds = concatenate_datasets([
    he_ds.shuffle(seed=42),
    nonhe_ds.shuffle(seed=42)
]).shuffle(seed=42)

# print(he_ds)

print("Balanced size:", len(balanced_ds))
print("HE:", sum(1 for x in balanced_ds if x["output"]=="HE"))
print("non-HE:", sum(1 for x in balanced_ds if x["output"]=="non-HE"))

# split
train_test = balanced_ds.train_test_split(test_size=0.2, seed=42)
train_ds, eval_ds = train_test["train"], train_test["test"]

# tokenize
def preprocess(ex):
    text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{json.dumps(ex['input'])}\n\n### Response:\n{ex['output']}"
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = train_ds.map(preprocess, remove_columns=dataset.column_names)
eval_ds = eval_ds.map(preprocess, remove_columns=dataset.column_names)

# collator
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

ft_model_path = "/raid/deeksha/mimic/trained_model/alpacare-he_balance/"   # or "./alpacare-he_balance/checkpoint-XXXX" if you want a specific step

# training args
args = TrainingArguments(
    output_dir=ft_model_path,
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

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer
)

if not os.path.exists(ft_model_path):
    trainer.train()

# load fine-tuned model instead of base
tokenizer = AutoTokenizer.from_pretrained(ft_model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(ft_model_path, torch_dtype=torch.float16, device_map="auto")

# ----------------------------
# 8. Inference example
# ----------------------------
test_example = df.sample(1).iloc[0]
prompt = f"""### Instruction:
Classify if this admission has hepatic encephalopathy (HE).

### Input:
{{
 "age": {test_example['anchor_age']},
 "bilirubin": {test_example.get('bilirubin', None)},
 "inr": {test_example.get('inr', None)},
 "albumin": {test_example.get('albumin', None)},
 "sodium": {test_example.get('sodium', None)},
 "creatinine": {test_example.get('creatinine', None)},
 "ammonia": {test_example.get('ammonia', None)},
 "meld_na": {test_example.get('meld_na', None)}
}}

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=50)
print("Model output:\n", tokenizer.decode(output[0], skip_special_tokens=True))
