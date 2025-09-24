import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your JSONL file
balanced_jsonl_path = "/raid/deeksha/mimic/he_notes_classification_balanced.jsonl"

# Load JSONL into a list of dicts
data = []
with open(balanced_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Transform into question/answer format
rows = []
for entry in data:
    instruction = entry.get("instruction", "").strip()
    input_data = entry.get("input", {})
    
    # Merge all input fields into a string
    merged_input = " ".join([f"{k}: {v}" for k, v in input_data.items() if v not in [None, "nan", "NaN"]])
    
    question = f"{instruction}\nGiven the clinical notes {merged_input}"
    answer = f"<output>{entry.get('output','').strip()}</output>"
    
    rows.append({"question": question, "answer": answer})

# Convert to DataFrame
df = pd.DataFrame(rows)

# Replace NaN with empty strings
df = df.fillna("")

# Train/test split (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV (no NaN, just empty)
train_df.to_csv("data/train.csv", index=False, na_rep="")
test_df.to_csv("data/test.csv", index=False, na_rep="")

print("✅ train.csv and test.csv have been saved successfully without NaN values!")


print(len(train_df)+ len(test_df))