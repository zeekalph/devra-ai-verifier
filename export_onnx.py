import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torchvision.models as models
import onnx
import gc

# Export Bert-Tiny (longer dummy, dynamic seq)
print("Exporting Bert-Tiny...")
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
model.eval()
dummy = tokenizer("Hello world" * 64, return_tensors="pt", max_length=512, truncation=True)["input_ids"]  # Seq=512
torch.onnx.export(
    model,
    dummy,
    "bert_tiny.onnx",
    export_params=True,
    opset_version=14,  # Stable
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}}  # Dynamic seq_len
)
onnx_model = onnx.load("bert_tiny.onnx")
onnx.checker.check_model(onnx_model)
print("Bert-Tiny exported")

# Export ResNet18 (static batch=1, H/W=224)
print("Exporting ResNet18...")
resnet = models.resnet18(pretrained=True)
resnet.eval()
dummy_img = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    resnet,
    dummy_img,
    "resnet18.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}}  # Dynamic batch only
)
onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)
print("ResNet18 exported")

gc.collect()
print("ONNX files ready – Commit to repo!")