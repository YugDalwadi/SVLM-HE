import os
import torch
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Paths
model_path = "/raid/deeksha/mimic/trained_models/multi_1st_run/checkpoint-39"
test_path = "data/test.csv"

# Reload fine-tuned model
max_seq_length = 2048
dtype = torch.float16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    attn_implementation="eager",
    device_map={"": 0},
)

# Apply chat template again
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Load test data
test_df = pd.read_csv(test_path)
test_df = test_df.head(10)

# Function to generate predictions
def generate_prediction(question, max_new_tokens=50):
    input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,  # get plain string, not tensor
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # greedy decoding
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# Run evaluation
y_true, y_pred = [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    question = row["question"]
    true_label = row["answer"].replace("<output>", "").replace("</output>", "").strip()
    
    generated_text = generate_prediction(question)
    print(generated_text)
    # Extract predicted label between <output> ... </output>
    if "<output>" in generated_text and "</output>" in generated_text:
        pred_label = generated_text.split("<output>")[1].split("</output>")[0].strip()
    else:
        pred_label = generated_text.strip().split()[-1]  # fallback
    
    y_true.append(true_label)
    y_pred.append(pred_label)

# Accuracy and report
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Classification Accuracy: {acc:.4f}\n")
print("📊 Detailed Report:\n", classification_report(y_true, y_pred))
