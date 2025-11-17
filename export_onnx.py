import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torchvision.models as models
import onnx
import gc

# Export Bert-Tiny (for text & relevance proxy)
print("Exporting Bert-Tiny...")
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
model.eval()
dummy = tokenizer("Hello world", return_tensors="pt")["input_ids"]  # Static seq=8
torch.onnx.export(
    model,
    dummy,
    "bert_tiny.onnx",
    export_params=True,
    opset_version=11,  # Stable, no conversion
    do_constant_folding=False,  # Disable to avoid naming bug
    input_names=['input_ids'],
    output_names=['logits']
)
onnx_model = onnx.load("bert_tiny.onnx")
onnx.checker.check_model(onnx_model)
print("Bert-Tiny exported")

# Export ResNet18 (for images)
print("Exporting ResNet18...")
resnet = models.resnet18(pretrained=True)
resnet.eval()
dummy_img = torch.randn(1, 3, 224, 224)  # Static batch=1, 224x224
torch.onnx.export(
    resnet,
    dummy_img,
    "resnet18.onnx",
    export_params=True,
    opset_version=11,  # Stable, no naming bug
    do_constant_folding=False,  # Disable to avoid key mismatch
    input_names=['input'],
    output_names=['output']
)
onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)
print("ResNet18 exported")

# Export SentenceTransformer (underlying BERT for MiniLM)
print("Exporting SentenceTransformer (MiniLM BERT)...")
from sentence_transformers import SentenceTransformer
sm = SentenceTransformer('all-MiniLM-L6-v2')
model = sm[0].auto_model  # Underlying BERT
tokenizer = sm[0].tokenizer
dummy = tokenizer("Hello world", return_tensors="pt")["input_ids"]  # Static seq=8
torch.onnx.export(
    model,
    dummy,
    "minilm_bert.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=False,  # Avoid naming bug
    input_names=['input_ids'],
    output_names=['last_hidden_state']
)
onnx_model = onnx.load("minilm_bert.onnx")
onnx.checker.check_model(onnx_model)
print("SentenceTransformer BERT exported")

gc.collect()
print("ONNX files ready – Commit to repo!")