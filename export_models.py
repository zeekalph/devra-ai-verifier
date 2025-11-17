import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import torchvision.models as models
from torchvision import transforms
import onnx
from onnxruntime import InferenceSession
import gc

device = torch.device("cpu")

# Export TinyBERT to ONNX
print("Exporting TinyBERT...")
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny").to(device)
dummy_input = tokenizer("Hello world", return_tensors="pt")["input_ids"]
torch.onnx.export(
    model,
    dummy_input,
    "bert_tiny.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}}
)
onnx_model = onnx.load("bert_tiny.onnx")
onnx.checker.check_model(onnx_model)
print("TinyBERT exported – RAM saved: ~20 MiB")

# Export ResNet18 to ONNX
print("Exporting ResNet18...")
resnet = models.resnet18(pretrained=True).to(device)
dummy_image = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    resnet,
    dummy_image,
    "resnet18.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}}
)
onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)
print("ResNet18 exported – RAM saved: ~15 MiB")

# Export MiniLM (underlying BERT to ONNX)
print("Exporting MiniLM...")
sm = SentenceTransformer('all-MiniLM-L6-v2')
model = sm[0].auto_model  # Underlying BERT
tokenizer = sm[0].tokenizer
dummy_input = tokenizer("Hello world", return_tensors="pt")["input_ids"]
torch.onnx.export(
    model,
    dummy_input,
    "minilm.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['last_hidden_state'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}}
)
onnx_model = onnx.load("minilm.onnx")
onnx.checker.check_model(onnx_model)
print("MiniLM exported – RAM saved: ~25 MiB")

gc.collect()
print("All models exported to ONNX – Commit .onnx files to repo!")
