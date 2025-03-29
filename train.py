import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer
from evaluate import load

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(torch.cuda.current_device())

data_files = {'train': 'train.csv', 'test': 'test.csv'}
dataset = load_dataset('csv', data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained('ai-forever/sbert_large_nlu_ru')

def tokenize_fn(data):
  return tokenizer(data['text'], padding='max_length', max_length=512, truncation=True)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

model_name = 'ai-forever/sbert_large_nlu_ru'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Initialize the custom model
# model = CustomBERTModel('ai-forever/sbert_large_nlu_ru', num_labels=2)

# for param in model.bert.parameters():
#   param.requires_grad = False

# for param in model.classifier.parameters():
#   param.requires_grad = True

# print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# =====================
# TRAIN
# =====================

training_args = TrainingArguments(
  output_dir="./results",           # Directory for saving model checkpoints
  evaluation_strategy="epoch",     # Evaluate at the end of each epoch
  save_strategy="epoch",
  learning_rate=5e-5,              # Start with a small learning rate
  per_device_train_batch_size=16,  # Batch size per GPU
  per_device_eval_batch_size=16,
  num_train_epochs=3,              # Number of epochs
  weight_decay=0.01,               # Regularization
  save_total_limit=2,              # Limit checkpoints to save space
  load_best_model_at_end=True,     # Automatically load the best checkpoint
  logging_dir="./logs",            # Directory for logs
  logging_steps=100,               # Log every 100 steps
  fp16=True                        # Enable mixed precision for faster training
)

metric = load('f1')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)
  
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset['train'],
  eval_dataset=tokenized_dataset['test'],
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics,
)

trainer.train()
