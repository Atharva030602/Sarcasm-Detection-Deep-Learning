"""
Run SBERT (Sentence-BERT) on the News Headlines Sarcasm Detection dataset.
Uses sentence-transformers to encode headlines and train a sarcasm classifier.
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
print("Loading dataset...")
headlines = []
labels = []
with open("datasets/news-headlines/Sarcasm_Headlines_Dataset.json", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        headlines.append(obj["headline"])
        labels.append(obj["is_sarcastic"])

print(f"Loaded {len(headlines)} headlines ({sum(labels)} sarcastic, {len(labels)-sum(labels)} non-sarcastic)")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    headlines, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Load SBERT model and encode sentences
print("Loading SBERT model (bert-base-nli-mean-tokens)...")
model = SentenceTransformer('bert-base-nli-mean-tokens')

print("Encoding training headlines...")
train_embeddings = model.encode(X_train, batch_size=32, show_progress_bar=True)

print("Encoding test headlines...")
test_embeddings = model.encode(X_test, batch_size=32, show_progress_bar=True)

# Train a classifier on the embeddings
print("Training logistic regression classifier...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(train_embeddings, y_train)

# Evaluate
y_pred = clf.predict(test_embeddings)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Sarcastic", "Sarcastic"]))
