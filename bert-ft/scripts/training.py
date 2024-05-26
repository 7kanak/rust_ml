from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer

import torch
print(torch.cuda.is_available())
dataset = load_dataset("yelp_review_full", split="train[:500]")
model_name = "albert/albert-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(500))


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5).to("cuda")
print(model.device)

training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=1, per_device_train_batch_size=4)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
)
trainer.train()
trainer.model.save_pretrained("./pytorch_model")


model = trainer.model.to("cpu")
from transformers.onnx import FeaturesManager, export as export_onnx
from pathlib import Path

feature = "sequence-classification"

model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

onnx_inputs, onnx_outputs = export_onnx(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=16,
        output=Path("albert2.onnx")
)
