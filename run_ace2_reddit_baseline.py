"""
ACE-2 baseline on reddit-sarc without contextual dialogue embedding.

This runner keeps the classic ACE-2 style setup:
- Contextual branch: BERT pooled embedding of target utterance only
- Affective branch: static lexicon features (EAISe-style 12-dim) + BiLSTM + MHA

It does NOT use dialogue-level contextual emotion embeddings.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="ACE-2 reddit baseline (static affect)")
    parser.add_argument("--dataset", type=str, default="datasets/reddit-sarc/sarcasm_data.json")
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reddit_samples(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []
    if isinstance(data, dict):
        records = data.values()
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unsupported dataset format")

    for obj in records:
        if not isinstance(obj, dict):
            continue
        text = str(obj.get("utterance", "")).strip()
        if not text:
            text = str(obj.get("response", "")).strip()
        label = obj.get("sarcasm", obj.get("is_sarcastic", None))
        if text and label is not None:
            texts.append(text)
            labels.append(int(bool(label)))

    if not texts:
        raise ValueError("No usable samples found in dataset")
    return texts, labels


def build_lexicons():
    nrc = {}
    with open("lexicons/nrc.txt", encoding="utf-8") as n:
        for line in n:
            line = line.rstrip()
            sp = line.split("\t")
            if len(sp) < 3:
                continue
            if sp[0] in nrc:
                nrc[sp[0]].append(int(sp[2]))
            else:
                nrc[sp[0]] = [int(sp[2])]

    emotion_sets = {}
    for emotion_name in ["anger", "fear", "sadness", "joy", "positive", "negative"]:
        emotion_sets[emotion_name] = set()
        with open(f"lexicons/{emotion_name}", encoding="utf-8") as f:
            for line in f:
                word = line.rstrip()
                if word and "_" not in word:
                    emotion_sets[emotion_name].add(word)
    return nrc, emotion_sets


def extract_affective_features(text, nrc, emotion_sets):
    tokens = word_tokenize(text.lower())
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
        if word in emotion_sets["anger"]:
            wd_anger += 1
        if word in emotion_sets["fear"]:
            wd_fear += 1
        if word in emotion_sets["sadness"]:
            wd_sadness += 1
        if word in emotion_sets["joy"]:
            wd_joy += 1
        if word in emotion_sets["positive"]:
            wd_pos += 1
        if word in emotion_sets["negative"]:
            wd_neg += 1

    return [
        nrc_ag,
        nrc_fe,
        nrc_jo,
        nrc_sa,
        nrc_pos,
        nrc_neg,
        wd_anger,
        wd_fear,
        wd_sadness,
        wd_joy,
        wd_pos,
        wd_neg,
    ]


class SarcasmDataset(Dataset):
    def __init__(self, input_ids, attention_masks, affective_features, labels, indices):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.affective_features = affective_features
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.tensor(self.input_ids[i], dtype=torch.long),
            torch.tensor(self.attention_masks[i], dtype=torch.long),
            torch.tensor(self.affective_features[i], dtype=torch.float),
            torch.tensor(self.labels[i], dtype=torch.long),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, num_units, num_heads=2, dropout_rate=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.k_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.v_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.output_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, queries, keys, values):
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)
        q_ = torch.cat(torch.chunk(q, self.num_heads, dim=2), dim=0)
        k_ = torch.cat(torch.chunk(k, self.num_heads, dim=2), dim=0)
        v_ = torch.cat(torch.chunk(v, self.num_heads, dim=2), dim=0)
        scores = torch.bmm(q_, k_.permute(0, 2, 1)) / (k_.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        weights = self.output_dropout(weights)
        outputs = torch.bmm(weights, v_)
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)
        return outputs + queries


class AffectiveBiLSTM_MHA(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_heads=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.multi_head_attn = MultiHeadAttention(hidden_dim, num_heads=num_heads, dropout_rate=dropout)
        self.label_embeddings = nn.Parameter(torch.randn(1, 2, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 6, 1)
        rnn_out, _ = self.bilstm(x)
        rnn_out = self.dropout(rnn_out)
        batch_size = rnn_out.size(0)
        label_embs = self.label_embeddings.expand(batch_size, -1, -1)
        attn_out = self.multi_head_attn(label_embs, rnn_out, rnn_out)
        return attn_out.reshape(batch_size, -1)


class ACE2BaselineModel(nn.Module):
    def __init__(self, affective_input_dim=12, affective_hidden_dim=64, num_heads=2, dropout=0.3):
        super().__init__()
        sbert = SentenceTransformer("bert-base-nli-mean-tokens")
        self.bert = sbert._modules["0"].auto_model
        del sbert

        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.bert.encoder.layer[-4:].parameters():
            p.requires_grad = True
        for p in self.bert.pooler.parameters():
            p.requires_grad = True

        self.affective = AffectiveBiLSTM_MHA(
            input_dim=affective_input_dim,
            hidden_dim=affective_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        combined_dim = 768 + (2 * affective_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, input_ids, attention_mask, affective_features):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs[1]
        affective_emb = self.affective(affective_features)
        combined = torch.cat([cls_output, affective_emb], dim=1)
        return self.classifier(combined)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("ACE-2 Baseline: Static Affective Features + Target Utterance BERT")
    print("=" * 70)

    print(f"\n[1/5] Loading dataset from: {args.dataset}")
    texts, labels = load_reddit_samples(args.dataset)
    print(f"  Loaded {len(texts)} samples ({sum(labels)} sarcastic, {len(labels)-sum(labels)} non-sarcastic)")

    print("\n[2/5] Building static lexicon affective features...")
    nrc, emotion_sets = build_lexicons()
    affective_features = np.array(
        [extract_affective_features(t, nrc, emotion_sets) for t in texts], dtype=np.float32
    )
    print(f"  Affective feature shape: {affective_features.shape}")

    print("\n[3/5] Tokenizing utterances...")
    sbert_tmp = SentenceTransformer("bert-base-nli-mean-tokens")
    tokenizer = sbert_tmp._modules["0"].tokenizer
    del sbert_tmp
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=args.max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]
    labels_arr = np.array(labels, dtype=np.int64)
    print(f"  Token shape: {input_ids.shape}, Attention mask shape: {attention_masks.shape}")

    print("\n[4/5] Train/test split...")
    indices = list(range(len(texts)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=labels_arr)
    train_loader = DataLoader(
        SarcasmDataset(input_ids, attention_masks, affective_features, labels_arr, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        SarcasmDataset(input_ids, attention_masks, affective_features, labels_arr, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    print("\n[5/5] Training baseline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = ACE2BaselineModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    bert_params = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "bert" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": 2e-5},
            {"params": other_params, "lr": 1e-3},
        ],
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_macro_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_ids, batch_mask, batch_affect, batch_y in train_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_affect = batch_affect.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_ids, batch_mask, batch_affect)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_ids, batch_mask, batch_affect, batch_y in test_loader:
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_affect = batch_affect.to(device)
                logits = model(batch_ids, batch_mask, batch_affect)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        if test_acc > best_acc:
            best_acc = test_acc
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

        print(
            f"  Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Macro F1: {macro_f1:.4f}"
        )

    print(f"\n{'=' * 70}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"{'=' * 70}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_ids, batch_mask, batch_affect, batch_y in test_loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_affect = batch_affect.to(device)
            logits = model(batch_ids, batch_mask, batch_affect)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    final_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")
    print(f"Final Macro F1: {final_macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))


if __name__ == "__main__":
    main()
