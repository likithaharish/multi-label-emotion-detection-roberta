# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# ==============================
# 2. DEVICE
# ==============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 3. LOAD DATA (80K)
# ==============================
df = pd.read_csv("go_emotions_dataset.csv")
df = df.sample(n=80000, random_state=42).reset_index(drop=True)

df = df.drop(columns=['id','author','subreddit','link_id','parent_id','created_utc','rater_id'], errors='ignore')

label_cols = df.columns.drop(['text'])
labels = df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

# ==============================
# 4. SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=42
)

# ==============================
# 5. TOKENIZER (RoBERTa)
# ==============================
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=100,
        return_tensors="pt"
    )

train_encodings = tokenize(X_train)
test_encodings = tokenize(X_test)

# ==============================
# 6. DATASET CLASS
# ==============================
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, y_train)
test_dataset = EmotionDataset(test_encodings, y_test)

# ==============================
# 7. MODEL (RoBERTa)
# ==============================
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=y_train.shape[1],
    problem_type="multi_label_classification"
)

model.to(device)

# ==============================
# 8. CUSTOM TRAINER (BCE LOSS)
# ==============================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ==============================
# 9. TRAINING ARGUMENTS
# ==============================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

# ==============================
# 10. METRICS
# ==============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.4).int().numpy()

    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average="micro"),
    }

# ==============================
# 11. TRAINER
# ==============================
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ==============================
# 12. TRAIN
# ==============================
print("Training Started...")
trainer.train()

# ==============================
# 13. THRESHOLD TUNING
# ==============================
print("\nFinding best threshold...")

predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids

probs = torch.sigmoid(torch.tensor(logits))

best_f1 = 0
best_threshold = 0

for t in np.arange(0.2, 0.6, 0.02):
    preds = (probs > t).int().numpy()
    f1 = f1_score(labels, preds, average="micro")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest Threshold: {best_threshold}")
print(f"Best F1 Score: {best_f1}")

# ==============================
# 14. SAVE MODEL
# ==============================
model.save_pretrained("emotion_roberta")
tokenizer.save_pretrained("emotion_roberta")

# ==============================
# 15. PREDICT FUNCTION
# ==============================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    preds = (probs > best_threshold)

    return {
        label_cols[i]: round(float(probs[i]), 3)
        for i in range(len(label_cols)) if preds[i]
    }

# ==============================
# 16. TEST
# ==============================
print("\nPrediction:", predict("I feel happy but also nervous"))

# ==============================
# 17. USER INPUT LOOP
# ==============================
model.eval()

print("\nModel is ready! Type your sentence to predict emotions.")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Enter text: ")

    if user_input.strip().lower() == "exit":
        print("Exiting...")
        break

    result = predict(user_input)

    if result:
        print("Predicted Emotions:", result)
    else:
        print("No strong emotion detected.")