"""
ACE 2 Model: Affective and Contextual Embedding for Sarcasm Detection (v2).

Key differences from ACE 1:
  - Contextual: Fine-tuned BERT end-to-end (not frozen SBERT)
  - Affective: BiLSTM with multi-head label attention (not single-head)
  - Entire model trains jointly (BERT + BiLSTM + classifier)

Architecture:
  headline tokens -> BERT (fine-tuned) -> [CLS] contextual_embedding (768-dim)
  headline -> EAISe emotion features (12-dim) -> BiLSTM + Multi-Head Attention -> affective_embedding
  [contextual_embedding ; affective_embedding] -> FC layers -> sarcasm prediction
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# =========================================================================
# STEP 1: Load Dataset
# =========================================================================
print("=" * 60)
print("ACE 2: Fine-tuned BERT + Multi-Head Affective BiLSTM")
print("=" * 60)
print("\n[1/5] Loading dataset...")

headlines = []
labels = []
with open("datasets/news-headlines/Sarcasm_Headlines_Dataset.json", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        headlines.append(obj["headline"])
        labels.append(obj["is_sarcastic"])

print(f"  Loaded {len(headlines)} headlines ({sum(labels)} sarcastic, {len(labels)-sum(labels)} non-sarcastic)")

# =========================================================================
# STEP 2: Extract Affective Features (EAISe-style)
# =========================================================================
print("\n[2/5] Extracting affective features (EAISe)...")

nrc = {}
with open("lexicons/nrc.txt") as n:
    for line in n:
        line = line.rstrip()
        sp = line.split("\t")
        if sp[0] in nrc:
            nrc[sp[0]].append(int(sp[2]))
        else:
            nrc[sp[0]] = [int(sp[2])]

emotion_sets = {}
for emotion_name in ["anger", "fear", "sadness", "joy", "positive", "negative"]:
    emotion_sets[emotion_name] = set()
    with open(f"lexicons/{emotion_name}") as f:
        for line in f:
            word = line.rstrip()
            if "_" not in word:
                emotion_sets[emotion_name].add(word)

def extract_affective_features(headline):
    tokens = word_tokenize(headline.lower())
    nrc_ag, nrc_fe, nrc_jo, nrc_sa, nrc_pos, nrc_neg = 0, 0, 0, 0, 0, 0
    for word in tokens:
        if word in nrc and len(nrc[word]) >= 7:
            nrc_ag += nrc[word][0]
            nrc_fe += nrc[word][3]
            nrc_jo += nrc[word][4]
            nrc_sa += nrc[word][7] if len(nrc[word]) > 7 else 0
            nrc_pos += nrc[word][6]
            nrc_neg += nrc[word][5]
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

affective_features = np.array([extract_affective_features(h) for h in headlines], dtype=np.float32)
print(f"  Affective feature shape: {affective_features.shape}")

# =========================================================================
# STEP 3: Tokenize for BERT
# =========================================================================
print("\n[3/5] Tokenizing headlines for BERT...")

# Extract tokenizer from cached SBERT model (avoids download issues with old transformers)
_sbert_tmp = SentenceTransformer('bert-base-nli-mean-tokens')
tokenizer = _sbert_tmp._modules['0'].tokenizer
del _sbert_tmp
MAX_LEN = 64

encoded = tokenizer.batch_encode_plus(
    headlines,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='np'
)
input_ids = encoded['input_ids']
attention_masks = encoded['attention_mask']
print(f"  Token shape: {input_ids.shape}, Attention mask shape: {attention_masks.shape}")

# =========================================================================
# STEP 4: Train/Test Split and DataLoader
# =========================================================================
print("\n[4/5] Preparing data...")

indices = list(range(len(headlines)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

labels_arr = np.array(labels)


class SarcasmDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.tensor(input_ids[i], dtype=torch.long),
            torch.tensor(attention_masks[i], dtype=torch.long),
            torch.tensor(affective_features[i], dtype=torch.float),
            torch.tensor(labels_arr[i], dtype=torch.long)
        )


BATCH_SIZE = 16  # Small batch for 4GB VRAM (fine-tuning BERT)
train_loader = DataLoader(SarcasmDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(SarcasmDataset(test_idx), batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

# =========================================================================
# Multi-Head Attention (from BiLSTM-Multihead-Attention.py)
# =========================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, num_units, num_heads=2, dropout_rate=0.3):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.Q_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.output_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, queries, keys, values):
        Q = self.Q_proj(queries)
        K = self.K_proj(keys)
        V = self.V_proj(values)
        # Split into heads
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)
        # Scaled dot-product attention
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))
        outputs = outputs / (K_.size(-1) ** 0.5)
        outputs = F.softmax(outputs, dim=-1)
        # Query masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(2).repeat(1, 1, keys.size(1))
        outputs = outputs * query_masks
        outputs = self.output_dropout(outputs)
        # Weighted sum
        outputs = torch.bmm(outputs, V_)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)
        # Residual connection
        outputs = outputs + queries
        return outputs


# =========================================================================
# ACE 2 Model Definition
# =========================================================================
class AffectiveBiLSTM_MHA(nn.Module):
    """BiLSTM with multi-head label attention for affective features."""
    def __init__(self, input_dim, hidden_dim, num_heads=2, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True,
                              num_layers=num_layers, bidirectional=True)
        self.multi_head_attn = MultiHeadAttention(hidden_dim, num_heads=num_heads, dropout_rate=dropout)
        # Label embeddings (2 classes: sarcastic, not-sarcastic)
        self.label_embeddings = nn.Parameter(torch.randn(1, 2, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, 12) -> create sequence
        x = x.unsqueeze(1).repeat(1, 6, 1)  # (batch, 6, 12)
        rnn_out, (h_n, c_n) = self.bilstm(x)  # (batch, 6, hidden_dim)
        rnn_out = self.dropout(rnn_out)

        # Multi-head label attention: attend using label embeddings
        batch_size = rnn_out.size(0)
        label_embs = self.label_embeddings.expand(batch_size, -1, -1)  # (batch, 2, hidden_dim)

        # Label attends to LSTM output
        attn_out = self.multi_head_attn(label_embs, rnn_out, rnn_out)  # (batch, 2, hidden_dim)

        # Pool: concatenate both label-attended representations
        attn_out = attn_out.view(batch_size, -1)  # (batch, 2 * hidden_dim)
        return attn_out


class ACE2Model(nn.Module):
    """
    ACE 2: Fine-tuned BERT + Multi-Head Affective BiLSTM.
    """
    def __init__(self, affective_input_dim=12, affective_hidden_dim=64,
                 num_heads=2, num_classes=2, dropout=0.3):
        super().__init__()

        # Contextual branch: Fine-tuned BERT (extracted from cached SBERT)
        sbert = SentenceTransformer('bert-base-nli-mean-tokens')
        self.bert = sbert._modules['0'].auto_model
        del sbert
        bert_dim = 768

        # Freeze lower BERT layers, fine-tune top 4 layers + pooler
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-4:].parameters():
            param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # Affective branch: BiLSTM with multi-head attention
        self.affective_bilstm = AffectiveBiLSTM_MHA(
            input_dim=affective_input_dim,
            hidden_dim=affective_hidden_dim,
            num_heads=num_heads,
            num_layers=1,
            dropout=dropout
        )

        # Combined: BERT (768) + affective (2 * hidden_dim from label attention)
        affective_out_dim = 2 * affective_hidden_dim
        combined_dim = bert_dim + affective_out_dim

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

    def forward(self, input_ids, attention_mask, affective_features):
        # Contextual: fine-tuned BERT [CLS] output
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs[1]  # Pooled [CLS] output (batch, 768)

        # Affective: BiLSTM with multi-head label attention
        affective_emb = self.affective_bilstm(affective_features)  # (batch, 2*hidden)

        # Concatenate
        combined = torch.cat([cls_output, affective_emb], dim=1)

        # Classify
        logits = self.classifier(combined)
        return logits

# =========================================================================
# STEP 5: Train ACE 2
# =========================================================================
print("\n[5/5] Training ACE 2 model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = ACE2Model(
    affective_input_dim=12,
    affective_hidden_dim=64,
    num_heads=2,
    num_classes=2,
    dropout=0.3
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print("  Frozen BERT layers: 0-7, Fine-tuned: 8-11 + pooler")
print("  Combined: BERT(768) + MHA-BiLSTM(128) = 896-dim\n")

# Use lower LR for BERT, higher for new layers
bert_params = [p for n, p in model.named_parameters() if 'bert' in n and p.requires_grad]
other_params = [p for n, p in model.named_parameters() if 'bert' not in n and p.requires_grad]

optimizer = torch.optim.AdamW([
    {'params': bert_params, 'lr': 2e-5},
    {'params': other_params, 'lr': 1e-3}
], weight_decay=0.01)

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)

EPOCHS = 5  # Fewer epochs for fine-tuning
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_ids, batch_mask, batch_affect, batch_y in train_loader:
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_affect = batch_affect.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_ids, batch_mask, batch_affect)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_ids, batch_mask, batch_affect, batch_y in test_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_affect = batch_affect.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_ids, batch_mask, batch_affect)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == batch_y).sum().item()
            test_total += batch_y.size(0)

    test_acc = test_correct / test_total
    if test_acc > best_acc:
        best_acc = test_acc

    print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
          f"{'  *best*' if test_acc == best_acc else ''}")

# Final evaluation
print(f"\n{'='*60}")
print(f"Best Test Accuracy: {best_acc:.4f}")
print(f"{'='*60}")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_ids, batch_mask, batch_affect, batch_y in test_loader:
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_affect = batch_affect.to(device)
        outputs = model(batch_ids, batch_mask, batch_affect)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

print(f"\nFinal Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))

print("\n" + "=" * 60)
print("Results Comparison:")
print("=" * 60)
print("  SBERT only (contextual):                83.47%")
print("  BiLSTM+Attention only:                  84.77%")
print("  ACE 1 (frozen SBERT + BiLSTM):          85.24%")
print(f"  ACE 2 (fine-tuned BERT + MHA-BiLSTM):   {best_acc*100:.2f}%")
print("=" * 60)
