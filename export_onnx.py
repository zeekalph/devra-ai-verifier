import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import torchvision.models as models
import onnx

# Export TinyBERT
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
dummy = tokenizer("Hello", return_tensors="pt")["input_ids"]
torch.onnx.export(model, dummy, "bert_tiny.onnx", opset_version=11, input_names=['input_ids'], output_names=['logits'], dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}})
onnx.checker.check_model(onnx.load("bert_tiny.onnx"))
print("TinyBERT ONNX exported")

# Export ResNet18
resnet = models.resnet18(pretrained=True)
dummy_img = torch.randn(1, 3, 224, 224)
torch.onnx.export(resnet, dummy_img, "resnet18.onnx", opset_version=11, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch'}})
onnx.checker.check_model(onnx.load("resnet18.onnx"))
print("ResNet18 ONNX exported")

# Export MiniLM (underlying BERT)
sm = SentenceTransformer('all-MiniLM-L6-v2')
model = sm[0].auto_model
tokenizer = sm[0].tokenizer
dummy = tokenizer("Hello", return_tensors="pt")["input_ids"]
torch.onnx.export(model, dummy, "minilm.onnx", opset_version=11, input_names=['input_ids'], output_names=['last_hidden_state'], dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}})
onnx.checker.check_model(onnx.load("minilm.onnx"))
print("MiniLM ONNX exported")

print("All ONNX files ready – Commit to repo!")
