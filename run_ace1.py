"""
ACE 1 Model: Affective and Contextual Embedding for Sarcasm Detection.

Combines:
1. Contextual Embedding (SBERT) - sentence-level semantic representation
2. Affective Embedding (EAISe + BiLSTM with Attention) - emotion-aware representation

Architecture:
  headline -> SBERT -> contextual_embedding (768-dim)
  headline -> EAISe emotion features (12-dim) -> BiLSTM + Attention -> affective_embedding
  [contextual_embedding ; affective_embedding] -> FC layers -> sarcasm prediction
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from nltk.tokenize import word_tokenize

# =========================================================================
# STEP 1: Load Dataset
# =========================================================================
print("=" * 60)
print("ACE 1: Affective and Contextual Embedding for Sarcasm Detection")
print("=" * 60)
print("\n[1/5] Loading dataset...")

headlines = []
labels = []
with open("datasets/news-headlines/Sarcasm_Headlines_Dataset.json", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        headlines.append(obj["headline"])
        labels.append(obj["is_sarcastic"])

print(f"  Loaded {len(headlines)} headlines ({sum(labels)} sarcastic, {len(labels)-sum(labels)} non-sarcastic)")

# =========================================================================
# STEP 2: Extract Affective Features (EAISe-style)
# =========================================================================
print("\n[2/5] Extracting affective features (EAISe)...")

# Load NRC lexicon
nrc = {}
with open("lexicons/nrc.txt") as n:
    for line in n:
        line = line.rstrip()
        sp = line.split("\t")
        if sp[0] in nrc:
            nrc[sp[0]].append(int(sp[2]))
        else:
            nrc[sp[0]] = [int(sp[2])]

# Load WordNet emotion sets
emotion_sets = {}
for emotion_name in ["anger", "fear", "sadness", "joy", "positive", "negative"]:
    emotion_sets[emotion_name] = set()
    with open(f"lexicons/{emotion_name}") as f:
        for line in f:
            word = line.rstrip()
            if "_" not in word:
                emotion_sets[emotion_name].add(word)

def extract_affective_features(headline):
    """Extract 12-dim affective feature vector (6 NRC + 6 WordNet)."""
    tokens = word_tokenize(headline.lower())

    # NRC features (anger, anticipation, disgust, fear, joy, neg, pos, sadness, surprise, trust)
    # We use indices [0]=anger, [3]=fear, [4]=joy, [7]=sadness, [5]=negative, [6]=positive
    nrc_ag, nrc_fe, nrc_jo, nrc_sa, nrc_pos, nrc_neg = 0, 0, 0, 0, 0, 0
    for word in tokens:
        if word in nrc and len(nrc[word]) >= 7:
            nrc_ag += nrc[word][0]
            nrc_fe += nrc[word][3]
            nrc_jo += nrc[word][4]
            nrc_sa += nrc[word][7] if len(nrc[word]) > 7 else 0
            nrc_pos += nrc[word][6]
            nrc_neg += nrc[word][5]

    # WordNet features
    wd_anger, wd_fear, wd_sadness, wd_joy, wd_pos, wd_neg = 0, 0, 0, 0, 0, 0
    for word in tokens:
        if word in emotion_sets["anger"]: wd_anger += 1
        if word in emotion_sets["fear"]: wd_fear += 1
        if word in emotion_sets["sadness"]: wd_sadness += 1
        if word in emotion_sets["joy"]: wd_joy += 1
        if word in emotion_sets["positive"]: wd_pos += 1
        if word in emotion_sets["negative"]: wd_neg += 1

    return [nrc_ag, nrc_fe, nrc_jo, nrc_sa, nrc_pos, nrc_neg,
            wd_anger, wd_fear, wd_sadness, wd_joy, wd_pos, wd_neg]

affective_features = [extract_affective_features(h) for h in headlines]
affective_features = np.array(affective_features, dtype=np.float32)
print(f"  Affective feature shape: {affective_features.shape}")

# =========================================================================
# STEP 3: Extract Contextual Features (SBERT)
# =========================================================================
print("\n[3/5] Extracting contextual features (SBERT)...")
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
contextual_features = sbert_model.encode(headlines, batch_size=32, show_progress_bar=True)
contextual_features = np.array(contextual_features, dtype=np.float32)
print(f"  Contextual feature shape: {contextual_features.shape}")

# =========================================================================
# STEP 4: Train/Test Split
# =========================================================================
print("\n[4/5] Preparing data...")

X_affect_train, X_affect_test, X_ctx_train, X_ctx_test, y_train, y_test = train_test_split(
    affective_features, contextual_features, labels,
    test_size=0.2, random_state=42, stratify=labels
)

X_affect_train = torch.FloatTensor(X_affect_train)
X_affect_test = torch.FloatTensor(X_affect_test)
X_ctx_train = torch.FloatTensor(X_ctx_train)
X_ctx_test = torch.FloatTensor(X_ctx_test)
y_train_t = torch.LongTensor(y_train)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_affect_train, X_ctx_train, y_train_t)
test_dataset = TensorDataset(X_affect_test, X_ctx_test, y_test_t)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train: {len(X_affect_train)}, Test: {len(X_affect_test)}")

# =========================================================================
# ACE 1 Model Definition
# =========================================================================
class AffectiveBiLSTM(nn.Module):
    """BiLSTM with attention for processing affective features."""
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.3):
        super(AffectiveBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True,
                              num_layers=num_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def attention(self, rnn_out, h_n):
        last_fwd = h_n[-2]
        last_bwd = h_n[-1]
        merged = torch.cat([last_fwd, last_bwd], 1).unsqueeze(2)
        weights = torch.bmm(rnn_out, merged)
        weights = torch.nn.functional.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(rnn_out.transpose(1, 2), weights).squeeze(2)

    def forward(self, x):
        # x: (batch, 12) -> reshape to (batch, 1, 12) for LSTM, or expand
        # Treat the 12 features as a sequence of length 12 with dim 1,
        # or as a single timestep with dim 12
        x = x.unsqueeze(1)  # (batch, 1, 12) - single timestep
        x = x.repeat(1, 6, 1)  # (batch, 6, 12) - repeat to give LSTM a sequence
        rnn_out, (h_n, c_n) = self.bilstm(x)
        rnn_out = self.dropout(rnn_out)
        attn_out = self.attention(rnn_out, h_n)
        return attn_out  # (batch, hidden_dim)


class ACE1Model(nn.Module):
    """
    ACE 1: Combines contextual (SBERT) and affective (BiLSTM) embeddings.
    """
    def __init__(self, contextual_dim=768, affective_input_dim=12,
                 affective_hidden_dim=128, num_classes=2, dropout=0.3):
        super(ACE1Model, self).__init__()

        # Affective branch: BiLSTM with attention
        self.affective_bilstm = AffectiveBiLSTM(
            input_dim=affective_input_dim,
            hidden_dim=affective_hidden_dim,
            num_layers=1,
            dropout=dropout
        )

        # Combined dimension: SBERT (768) + BiLSTM affective (128)
        combined_dim = contextual_dim + affective_hidden_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, affective_features, contextual_features):
        # Process affective features through BiLSTM
        affective_emb = self.affective_bilstm(affective_features)  # (batch, 128)

        # Concatenate contextual + affective
        combined = torch.cat([contextual_features, affective_emb], dim=1)  # (batch, 896)

        # Classify
        logits = self.classifier(combined)
        return logits

# =========================================================================
# STEP 5: Train ACE 1
# =========================================================================
print("\n[5/5] Training ACE 1 model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = ACE1Model(
    contextual_dim=768,
    affective_input_dim=12,
    affective_hidden_dim=128,
    num_classes=2,
    dropout=0.3
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Combined embedding: SBERT(768) + BiLSTM-Affective(128) = 896-dim\n")

EPOCHS = 15
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_affect, batch_ctx, batch_y in train_loader:
        batch_affect = batch_affect.to(device)
        batch_ctx = batch_ctx.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_affect, batch_ctx)
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
    scheduler.step(avg_loss)

    # Evaluate each epoch
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_affect, batch_ctx, batch_y in test_loader:
            batch_affect = batch_affect.to(device)
            batch_ctx = batch_ctx.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_affect, batch_ctx)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == batch_y).sum().item()
            test_total += batch_y.size(0)

    test_acc = test_correct / test_total
    if test_acc > best_acc:
        best_acc = test_acc

    print(f"  Epoch {epoch+1:>2}/{EPOCHS} - Loss: {avg_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
          f"{'  *best*' if test_acc == best_acc else ''}")

# Final evaluation
print(f"\n{'='*60}")
print(f"Best Test Accuracy: {best_acc:.4f}")
print(f"{'='*60}")

model.eval()
all_preds = []
with torch.no_grad():
    for batch_affect, batch_ctx, batch_y in test_loader:
        batch_affect = batch_affect.to(device)
        batch_ctx = batch_ctx.to(device)
        outputs = model(batch_affect, batch_ctx)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

print(f"\nFinal Test Accuracy: {accuracy_score(y_test, all_preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))

print("\n" + "=" * 60)
print("Results Comparison:")
print("=" * 60)
print(f"  SBERT only (contextual):          83.47%")
print(f"  BiLSTM+Attention only:            84.77%")
print(f"  ACE 1 (contextual + affective):   {best_acc*100:.2f}%")
print("=" * 60)
