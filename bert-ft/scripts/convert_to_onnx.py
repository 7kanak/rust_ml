import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.onnx import FeaturesManager, export as export_onnx
from pathlib import Path

model_id = "albert/albert-base-v2"
feature = "sequence-classification"

model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("the movie was wonderful but i prefer tv seriesh", return_tensors="pt")
# print(inputs)
with torch.no_grad():
    outputs = model(**inputs)
print("\n---- \n",outputs["logits"])

# model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
# onnx_config = model_onnx_config(model.config)


# onnx_inputs, onnx_outputs = export_onnx(
#         preprocessor=tokenizer,
#         model=model,
#         config=onnx_config,
#         opset=16,
#         output=Path("albert.onnx")
# )


# torch.onnx.export(
#     model, 
#     tuple(inputs.values()),
#     f="albert.onnx",  
#     input_names=['input_ids', 'attention_mask'], 
#     output_names=['logits'], 
#     dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
#                   'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
#                   'logits': {0: 'batch_size', 1: 'sequence'}}, 
#     do_constant_folding=True, 
#     opset_version=16, 
#     verbose=True
# )