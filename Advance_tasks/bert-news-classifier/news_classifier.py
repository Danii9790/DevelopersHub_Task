from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import torch

dataset = load_dataset("ag_news")
labels = dataset['train'].features['label'].names


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'].shuffle(seed=42).select(range(5000)),
    eval_dataset=encoded_dataset['test'].shuffle(seed=42).select(range(1000)),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
