import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet18
import torch
import gc

cache_dir = ".cache/huggingface"  
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir

device = torch.device("cpu")

print("Caching TinyBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", cache_dir=cache_dir)
gc.collect()

print("Caching TinyBERT model...")
model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny", cache_dir=cache_dir).to(device)
gc.collect()

print("Caching MiniLM...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=os.path.join(cache_dir, "sentence_transformers"))
gc.collect()

print("Caching ResNet-18...")
resnet = resnet18(pretrained=True).to(device)
gc.collect()

print("Cache complete! Folder size: ~100 MiB. Commit .cache/ to repo.")
print("Models ready: TinyBERT (29 MiB), MiniLM (80 MiB), ResNet-18 (45 MiB).")