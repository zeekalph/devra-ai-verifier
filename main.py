from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import zipfile
import pandas as pd
import numpy as np
import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="AI Dataset Verifier")

device = torch.device("cpu")

# Lazy load globals
_tokenizer = None
_model = None
_sentence_model = None
_resnet = None
_transform = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")  # 4.4M params, ~29 MiB
    return _tokenizer

def get_model():
    global _model
    if _model is None:
        _model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny").to(device)
        _model.eval()
    return _model

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # 22M params, ~80 MiB
    return _sentence_model

def get_resnet():
    global _resnet, _transform
    if _resnet is None:
        _resnet = models.resnet18(pretrained=True).to(device)  # 11.7M params, ~45 MiB
        _resnet.eval()
        _transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return _resnet, _transform

class Response(BaseModel):
    scores: Dict[str, int]
    status: str
    issues: List[str] = []

def score_text(texts: List[str], desc: str = None):
    if not texts:
        return {"quality": 0, "completeness": 0, "consistency": 0, "relevance": 50}
    tokenizer = get_tokenizer()
    model = get_model()
    perps = []
    for t in texts[:2]:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
            perps.append(torch.exp(loss).item())
        del enc; gc.collect()
    quality = max(0, min(100, 100 - np.mean(perps) * 2))
    relevance = 50
    if desc and texts:
        sm = get_sentence_model()
        e1 = sm.encode(desc, convert_to_tensor=True)
        e2 = sm.encode(texts[:3], convert_to_tensor=True)
        sim = util.cos_sim(e1, e2).mean().item()
        relevance = int((sim + 1) * 50)
    return {
        "quality": int(quality),
        "completeness": 100 if len(texts) >= 2 else 50,
        "consistency": 90,
        "relevance": relevance
    }

@app.post("/verify", response_model=Response)
async def verify(file: UploadFile = File(...), description: str = Form(None)):
    content = await file.read()
    texts = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            for n in z.namelist():
                if n.endswith(('.csv', '.txt')):
                    data = z.read(n)
                    if n.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(data))
                        texts.extend(df.astype(str).values.flatten().tolist())
                    else:
                        texts.append(data.decode())
    except:
        try:
            texts = [content.decode()]
        except:
            pass
    scores = score_text(texts, description)
    status = "VERIFIED" if scores["quality"] >= 60 else "FAILED"
    return Response(scores=scores, status=status, issues=[])

@app.get("/")
def home():
    return {"message": "AI Verifier API – Use /docs"}
