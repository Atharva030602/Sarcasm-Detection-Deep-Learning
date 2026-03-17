"""
Run BiLSTM with Attention on the News Headlines Sarcasm Detection dataset.
Uses the LSTMAttention model from BiLSTM-Multihead-Attention.py.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# ---- Load Dataset ----
print("Loading dataset...")
headlines = []
labels = []
with open("datasets/news-headlines/Sarcasm_Headlines_Dataset.json", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        headlines.append(obj["headline"].lower())
        labels.append(obj["is_sarcastic"])

print(f"Loaded {len(headlines)} headlines ({sum(labels)} sarcastic, {len(labels)-sum(labels)} non-sarcastic)")

# ---- Build Vocabulary ----
print("Building vocabulary...")
all_words = []
for h in headlines:
    all_words.extend(h.split())

word_counts = Counter(all_words)
# Keep words that appear at least 2 times
vocab = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.most_common()) if count >= 2}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# ---- Tokenize and Pad ----
MAX_LEN = 30  # Headlines are short

def encode_headline(headline, vocab, max_len):
    tokens = headline.split()
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    # Pad to max_len
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

encoded = [encode_headline(h, vocab, MAX_LEN) for h in headlines]

# ---- Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    encoded, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train = torch.LongTensor(X_train)
X_test = torch.LongTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test_tensor)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ---- Define Model (from BiLSTM-Multihead-Attention.py) ----
class LSTMAttention(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size, num_layers=1, dropout=0.3):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = torch.cuda.is_available()
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True,
                              num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(dropout)

    def attention(self, rnn_out, state):
        # state shape: (num_layers*2, batch, hidden_dim//2)
        # Take only the last layer's forward and backward hidden states
        # For 2-layer BiLSTM: indices [-2] (fwd) and [-1] (bwd)
        last_fwd = state[-2]  # (batch, hidden_dim//2)
        last_bwd = state[-1]  # (batch, hidden_dim//2)
        merged_state = torch.cat([last_fwd, last_bwd], 1)  # (batch, hidden_dim)
        merged_state = merged_state.unsqueeze(2)  # (batch, hidden_dim, 1)
        weights = torch.bmm(rnn_out, merged_state)  # (batch, seq_len, 1)
        weights = torch.nn.functional.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, X):
        batch_size = X.size(0)
        embedded = self.word_embeddings(X)
        embedded = self.dropout(embedded)

        h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2)
        if self.use_gpu:
            h0, c0 = h0.cuda(), c0.cuda()

        rnn_out, (h_n, c_n) = self.bilstm(embedded, (h0, c0))
        attn_out = self.attention(rnn_out, h_n)
        attn_out = self.dropout(attn_out)
        logits = self.hidden2label(attn_out)
        return logits

# ---- Initialize Model ----
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LABEL_SIZE = 2
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMAttention(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    label_size=LABEL_SIZE,
    num_layers=NUM_LAYERS,
    dropout=0.3
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Config: embed_dim={EMBEDDING_DIM}, hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS}, epochs={EPOCHS}\n")

# ---- Training Loop ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

# ---- Evaluation ----
print("\nEvaluating on test set...")
model.eval()
all_preds = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

acc = accuracy_score(y_test, all_preds)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))
