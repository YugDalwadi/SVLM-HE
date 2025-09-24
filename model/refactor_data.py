from datasets import load_dataset

def split_merged(row):
    q_end = row["merged"].find(".")
    question = row["merged"][:q_end+1]
    context = row["merged"][q_end+1:].strip()
    return {"question": question, "context": context, "answer": row["answer"]}


dataset = load_dataset('csv',
                       data_files={'train': 'ydalwadi/data/train.csv',
                                   'test': 'ydalwadi/data/train.csv'})

# Apply this to your dataset before tokenization
dataset = dataset.map(split_merged)
