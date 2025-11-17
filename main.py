from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from typing import List, Dict
import onnxruntime as ort  # ONNXRuntime for inference
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io
import zipfile
import pandas as pd
import numpy as np
import gc
import os

app = FastAPI(title="AI Dataset Verifier")

# ONNX providers (CPU only for Render)
providers = ['CPUExecutionProvider']

# Lazy load ONNX sessions
_bert_session = None
_minilm_session = None
_resnet_session = None
_tokenizer = None
_sentence_model = None

def get_bert_session():
    global _bert_session
    if _bert_session is None:
        _bert_session = ort.InferenceSession("bert_tiny.onnx", providers=providers)
    return _bert_session

def get_minilm_session():
    global _minilm_session
    if _minilm_session is None:
        _minilm_session = ort.InferenceSession("minilm.onnx", providers=providers)
    return _minilm_session

def get_resnet_session():
    global _resnet_session
    if _resnet_session is None:
        _resnet_session = ort.InferenceSession("resnet18.onnx", providers=providers)
    return _resnet_session

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    return _tokenizer

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Keep for embedding (ONNX for inference only)
    return _sentence_model

class Response(BaseModel):
    scores: Dict[str, int]
    status: str
    issues: List[str] = []

def score_text(texts: List[str], desc: str = None):
    if not texts:
        return {"quality": 0, "completeness": 0, "consistency": 0, "relevance": 50}
    tokenizer = get_tokenizer()
    session = get_bert_session()
    perps = []
    for t in texts[:2]:
        enc = tokenizer(t, return_tensors="pt")["input_ids"].numpy()
        outputs = session.run(None, {"input_ids": enc})
        logits = outputs[0]
        loss = np.log(np.sum(np.exp(logits))) - np.log(logits.shape[-1])  # Approx perplexity
        perps.append(np.exp(loss))
    quality = max(0, min(100, 100 - np.mean(perps) * 2))
    relevance = 50
    if desc and texts:
        sm = get_sentence_model()
        e1 = sm.encode(desc, convert_to_tensor=True)
        e2 = sm.encode(texts[:3], convert_to_tensor=True)
        sim = util.cos_sim(e1, e2).mean().item()
        relevance = int((sim + 1) * 50)
    gc.collect()
    return {
        "quality": int(quality),
        "completeness": 100 if len(texts) >= 2 else 50,
        "consistency": 90,
        "relevance": relevance
    }

def score_image(images: List[bytes]):
    if not images:
        return {"quality": 0, "completeness": 0, "consistency": 0, "relevance": 50}
    session = get_resnet_session()
    confidences = []
    for img_bytes in images[:2]:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        tensor = transforms.ToTensor()(img).unsqueeze(0).numpy()
        outputs = session.run(None, {"input": tensor})
        probs = torch.nn.functional.softmax(torch.tensor(outputs[0]), dim=1).numpy()
        top5 = np.mean(np.sort(probs[0])[-5:])
        confidences.append(top5)
    quality = int(np.mean(confidences) * 100)
    gc.collect()
    return {
        "quality": quality,
        "completeness": 100 if len(images) >= 2 else 50,
        "consistency": 90,
        "relevance": quality
    }

@app.post("/verify", response_model=Response)
async def verify(file: UploadFile = File(...), description: str = Form(None)):
    content = await file.read()
    texts = []
    images = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            for n in z.namelist():
                data = z.read(n)
                if n.endswith(('.csv', '.txt')):
                    if n.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(data))
                        texts.extend(df.astype(str).values.flatten().tolist())
                    else:
                        texts.append(data.decode())
                elif n.endswith(('.png', '.jpg', '.jpeg')):
                    images.append(data)
    except:
        try:
            texts = [content.decode()]
        except:
            pass
    text_scores = score_text(texts, description)
    image_scores = score_image(images)
    # Weighted average (60% text, 40% image)
    scores = {k: int(0.6 * text_scores[k] + 0.4 * image_scores[k]) for k in text_scores}
    status = "VERIFIED" if scores["quality"] >= 60 else "FAILED"
    gc.collect()
    return Response(scores=scores, status=status, issues=[])

@app.get("/")
def home():
    return {"message": "AI Verifier API – Use /docs"}
