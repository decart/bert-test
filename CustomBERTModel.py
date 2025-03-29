import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class CustomBERTModel(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

        for param in self.bert.bert.parameters():
          param.requires_grad = False

        for param in self.bert.classifier.parameters():
          param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels) 
        pooled_output = self.dropout(output[1])  # Applying dropout
        logits = self.fc(pooled_output)  # Adding a fully connected layer
        return logits
