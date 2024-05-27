from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

s = time.time()
model = AutoModelForSequenceClassification.from_pretrained("/home/kanak/Documents/dev/rust_ml/bert-ft/scripts/pytorch_model").cuda()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")



# Example input text
with open("/home/kanak/Documents/dev/rust_ml/trf-ort/data/test.txt") as f:
    texts = f.readlines()

# for text in texts:
inputs = tokenizer(texts, return_tensors="pt", padding=True)
inputs = {key: value.cuda() for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
print( outputs.logits)

print(time.time() - s)