from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import requests
import ipfshttpclient
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import base64
import json
import random
import torch
import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import zipfile
import pandas as pd
from io import BytesIO, StringIO
import mimetypes
from PIL import Image
import io
import pyarrow.parquet as pq
import openpyxl
import numpy as np
from typing import List, Dict, Any
import gc
from sentence_transformers import SentenceTransformer, util


app = FastAPI(
    title="AI Dataset Verifier (pre-encryption)",
    description="Runs BERT + ResNet-50, flags bad data, returns 4 scores.",
    version="1.0.0",
)


device = torch.device("cpu")      


distil_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distil_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
distil_model.eval()
gc.collect()

sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
gc.collect()


resnet = resnet18(pretrained=True).to(device)
resnet.eval()
resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
gc.collect()


class Issue(BaseModel):
    file: str
    type: str                     # "missing", "duplicate", "outlier", "text", "image"
    details: str

class VerifyRequest(BaseModel):
    ipfsCid: str
    tempDecryptionKey: str


class VerifyResponse(BaseModel):
    scores: dict[str, int]  
    status: str  
    issues: List[Issue] = []


BAD_DATA_CONFIG = {
    "missing_threshold": 0.30,
    "duplicate_threshold": 0.10,
    "z_score_threshold": 3.0,
    "text_min_len": 1,
    "text_max_len": 500,
    "image_min_size": 50,      # px on smallest side
    "image_max_black": 0.95,   # fraction of black pixels
}


def process_file(file_name: str, file_bytes: bytes, file_type: str, text_data: list, image_data: list):
    """Classify and parse file based on type."""
    if not file_type:
        file_type = "unknown"

    if file_type.startswith('image/'):
        image_data.append(file_bytes)
    elif file_type in ['application/json', 'text/csv', 'application/octet-stream', 'text/plain']:
        # Handle JSON, CSV, Parquet, Excel, Text
        try:
            if file_type == 'application/json':
                data = json.loads(file_bytes.decode('utf-8'))
                text_data.extend(extract_text_from_dict(data))
            elif file_type == 'text/csv':
                df = pd.read_csv(BytesIO(file_bytes))
                text_data.extend(df.select_dtypes(include=['object']).values.flatten().astype(str))
            elif file_name.endswith('.parquet'):
                df = pq.read_table(BytesIO(file_bytes)).to_pandas()
                text_data.extend(df.select_dtypes(include=['object']).values.flatten().astype(str))
            elif file_name.endswith('.xlsx'):
                df = pd.read_excel(BytesIO(file_bytes), engine='openpyxl')
                text_data.extend(df.select_dtypes(include=['object']).values.flatten().astype(str))
            elif file_type == 'text/plain':
                text_data.append(file_bytes.decode('utf-8'))
        except Exception as e:
            print(f"Error parsing {file_name}: {e}")



def compute_relevance_score(description: str, texts: List[str]) -> int:
    if not description or not texts:
        return 50  # neutral

    desc_emb = sentence_model.encode(description, convert_to_tensor=True)
    content_embs = sentence_model.encode(texts[:10], convert_to_tensor=True)  # limit
    similarities = util.cos_sim(desc_emb, content_embs)[0]
    avg_sim = similarities.mean().item()
    # Map cosine similarity [-1,1] → [0,100]
    score = int((avg_sim + 1) * 50)
    return max(0, min(100, score))


def extract_text_from_dict(data, max_depth=3, current_depth=0):
    """Recursively extract strings from nested JSON."""
    texts = []
    if current_depth > max_depth:
        return texts
    if isinstance(data, str):
        texts.append(data)
    elif isinstance(data, (list, tuple)):
        for item in data:
            texts.extend(extract_text_from_dict(item, max_depth, current_depth + 1))
    elif isinstance(data, dict):
        for value in data.values():
            texts.extend(extract_text_from_dict(value, max_depth, current_depth + 1))
    return texts


def zero_scores() -> dict:
    return {"quality": 0, "completeness": 0, "consistency": 0, "relevance": 0}

def flag_bad_data(
    texts: List[str],
    images: List[bytes],
    dfs: List[pd.DataFrame],
    names: List[str],
) -> List[Issue]:
    issues: List[Issue] = []
    cfg = BAD_DATA_CONFIG

    # ----- Tabular -----
    for df, fname in zip(dfs, names):
        miss = df.isna().mean()
        bad_cols = miss[miss > cfg["missing_threshold"]].index.tolist()
        if bad_cols:
            issues.append(Issue(file=fname, type="missing",
                                details=f"Columns >{cfg['missing_threshold']*100:.0f}% NaN: {bad_cols}"))

        dup_frac = df.duplicated().mean()
        if dup_frac > cfg["duplicate_threshold"]:
            issues.append(Issue(file=fname, type="duplicate",
                                details=f"{dup_frac*100:.1f}% duplicate rows"))

        for col in df.select_dtypes(include="number"):
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            if (z > cfg["z_score_threshold"]).any():
                issues.append(Issue(file=fname, type="outlier",
                                    details=f"{(z>cfg['z_score_threshold']).sum()} outliers in '{col}'"))

    # ----- Text -----
    for txt, fname in zip(texts, names):
        if not txt.strip():
            issues.append(Issue(file=fname, type="text", details="Empty"))
            continue
        if len(txt) < cfg["text_min_len"]:
            issues.append(Issue(file=fname, type="text", details="Too short"))
        if len(txt) > cfg["text_max_len"]:
            issues.append(Issue(file=fname, type="text", details="Too long"))
        if sum(c.isalnum() or c.isspace() for c in txt) / len(txt) < 0.1:
            issues.append(Issue(file=fname, type="text", details="Gibberish"))

    # ----- Images -----
    for img_bytes, fname in zip(images, names):
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = img.size
            if min(w, h) < cfg["image_min_size"]:
                issues.append(Issue(file=fname, type="image", details="Too small"))
                continue
            arr = np.array(img)
            black_frac = np.mean(np.all(arr == 0, axis=-1))
            if black_frac > cfg["image_max_black"]:
                issues.append(Issue(file=fname, type="image", details="Mostly black"))
        except Exception as e:
            issues.append(Issue(file=fname, type="image", details=f"Corrupt: {e}"))

    return issues


def combine_scores(text_s: dict, img_s: dict, has_text: bool, has_img: bool) -> dict:
    if has_text and has_img:
        return {k: int(0.6 * text_s[k] + 0.4 * img_s[k]) for k in text_s}
    return text_s if has_text else img_s



def ai_verify_data(raw_bytes: bytes, description: str = None) -> tuple[dict, str, List[Issue]]:
    issues: List[Issue] = []

    # ----- 1. Collect raw buckets -----
    text_data: List[str] = []
    image_data: List[bytes] = []
    tabular_data: List[pd.DataFrame] = []
    file_names: List[str] = []

    if zipfile.is_zipfile(io.BytesIO(raw_bytes)):
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
            for name in zf.namelist():
                fbytes = zf.read(name)
                ftype, _ = mimetypes.guess_type(name)

                if ftype not in [
                    "application/json", "text/csv", "application/octet-stream",
                    "text/plain",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "image/png", "image/jpeg",
                ]:
                    issues.append(Issue(
                        file=name, type="unsupported",
                        details="Supported: JSON/CSV/Parquet/Excel/TXT/PNG/JPEG"
                    ))
                    continue

                file_names.append(name)
                if ftype and ftype.startswith("image/"):
                    image_data.append(fbytes)
                elif ftype == "application/json":
                    text_data.extend(extract_text_from_dict(json.loads(fbytes.decode("utf-8"))))
                elif ftype == "text/csv":
                    tabular_data.append(pd.read_csv(io.BytesIO(fbytes)))
                elif name.endswith(".parquet"):
                    tabular_data.append(pq.read_table(io.BytesIO(fbytes)).to_pandas())
                elif name.endswith(".xlsx"):
                    tabular_data.append(pd.read_excel(io.BytesIO(fbytes), engine="openpyxl"))
                elif ftype == "text/plain":
                    text_data.append(fbytes.decode("utf-8"))
    else:
        name = "single_file"
        ftype, _ = mimetypes.guess_type(name)
        file_names.append(name)
        if ftype and ftype.startswith("image/"):
            image_data.append(raw_bytes)
        else:
            text_data.append(raw_bytes.decode("utf-8"))

    # ----- 2. Flag bad data -----
    flag_issues = flag_bad_data(text_data, image_data, tabular_data, file_names)
    issues.extend(flag_issues)

    # ----- 3. Score with description context -----
    text_scores = score_text_data(text_data, description=description) if text_data else zero_scores()
    img_scores = score_image_data(image_data) if image_data else zero_scores()

    final_scores = combine_scores(
        text_s=text_scores, img_s=img_scores,
        has_text=bool(text_data), has_img=bool(image_data)
    )

    status = "VERIFIED" if final_scores["quality"] >= 50 else "FAILED"
    return final_scores, status, issues


def score_text_data(texts: List[str], description: str = None) -> dict:
    if not texts:
        return zero_scores()

    perplexities = []
    for txt in texts[:5]:                     # still limit for speed
        enc = distil_tokenizer(txt, return_tensors="pt",
                               truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = distil_model(**enc, labels=enc["input_ids"])
            loss = out.loss
            perplexities.append(torch.exp(loss).item())

    avg_perp = np.mean(perplexities) if perplexities else float("inf")
    quality = max(0, min(100, 100 - avg_perp * 2))
    completeness = 100 if len(texts) >= 5 else len(texts) * 20
    consistency = max(0, min(100, 100 - np.std(perplexities) * 10))
    relevance = compute_relevance_score(description, texts)

    return {
        "quality": int(quality),
        "completeness": int(completeness),
        "consistency": int(consistency),
        "relevance": relevance,
    }


def score_image_data(images: List[bytes]) -> dict:
    if not images:
        return zero_scores()

    confidences = []
    for img_bytes in images[:3]:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensor = resnet_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = resnet(tensor)                     # <-- ResNet-18
                probs = torch.nn.functional.softmax(out[0], dim=0)
                top5 = probs.topk(5).values.sum().item() / 5
                confidences.append(top5)
        except Exception:
            continue

    avg_conf = np.mean(confidences) if confidences else 0.0
    quality = int(avg_conf * 100)
    completeness = 100 if len(images) >= 3 else len(images) * 33
    consistency = max(0, min(100, 100 - np.std(confidences) * 50))
    relevance = quality
    return { "quality": quality, "completeness": completeness,
             "consistency": consistency, "relevance": relevance }


@app.post("/verify", response_model=VerifyResponse)
async def verify_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None), 
):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    scores, status, issues = ai_verify_data(raw_bytes, description=description)

    return VerifyResponse(scores=scores, status=status, issues=issues)

#  Health check
@app.get("/")
def root():
    return {"message": "AI Verifier (pre-encryption) is up!"}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)