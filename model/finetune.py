from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator
import torch

dataset = load_dataset('csv',
                       data_files={'train': 'data/train.csv',
                                   'test': 'data/test.csv'})


def split_merged(row):
    q_end = row["question"].find(".")
    question = row["question"][:q_end+1]
    context = row["question"][q_end+1:].strip()
    return {"question": question, "context": context, "answer": row["answer"]}


dataset = dataset.map(split_merged)


def tokenize(batch):
    return tokenizer(
        batch["question"],
        batch["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

args = TrainingArguments(
    output_dir="finetune-BERT",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DefaultDataCollator()

tokenized_dataset = dataset.map(tokenize, batched=True)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],  # type:ignore
    eval_dataset=tokenized_dataset["test"],  # type:ignore
    data_collator=data_collator,
    processing_class=tokenizer
)

trainer.train()
