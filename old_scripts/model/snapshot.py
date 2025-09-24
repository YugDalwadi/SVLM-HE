import sys
import pandas
from huggingface_hub import login, snapshot_download

# Login to Hugging Face
huggingface_token = ""  # Replace with your actual token
login(token=huggingface_token)

model_name = 'medalpaca/medalpaca-7b'

local_dir = "/raid/deeksha/mimic/models/medalpaca/medalpaca-7b"

snapshot_download(model_name, local_dir=local_dir, allow_patterns=["*"])
print(f"Model downloaded to {local_dir}")
