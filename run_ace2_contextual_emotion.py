"""
ACE 2.1: Dialogue-aware sarcasm detection with contextual emotion embeddings.

This script upgrades the affective branch from static lexicon counts to dynamic,
contextual emotion embeddings learned from utterance representations.

Supported input formats:
1) News headlines JSONL (default):
   {"headline": "...", "is_sarcastic": 0/1}
2) Dialogue JSONL (optional custom file):
   {
     "context": ["turn1", "turn2", ...],
     "response": "target turn text",
     "is_sarcastic": 0/1
   }

Run example:
  /path/to/python run_ace2_contextual_emotion.py \
    --dataset datasets/news-headlines/Sarcasm_Headlines_Dataset.json
"""

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="ACE 2.1 contextual-emotion trainer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/news-headlines/Sarcasm_Headlines_Dataset.json",
        help="Path to JSONL dataset",
    )
    parser.add_argument("--max-turns", type=int, default=4, help="Max dialogue turns per sample")
    parser.add_argument("--max-len", type=int, default=96, help="Max tokens per turn")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bert-lr", type=float, default=2e-5)
    parser.add_argument("--other-lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "sarcastic", "sarc", "yes", "true"}:
            return 1
        if v in {"0", "not_sarcastic", "non_sarcastic", "no", "false"}:
            return 0
    return None


def _extract_turns_and_label(obj):
    label_keys = ["is_sarcastic", "label", "sarcasm", "sarc"]
    label = None
    for key in label_keys:
        if key in obj:
            label = _normalize_label(obj[key])
            if label is not None:
                break
    if label is None:
        return None

    if "headline" in obj:
        text = str(obj.get("headline", "")).strip()
        if not text:
            return None
        return [text], label

    context = obj.get("context", [])
    if isinstance(context, str):
        context = [context]
    context = [str(x).strip() for x in context if str(x).strip()]
    context_speakers = obj.get("context_speakers", [])
    if isinstance(context_speakers, str):
        context_speakers = [context_speakers]
    context_speakers = [str(x).strip() for x in context_speakers]

    response = ""
    for key in ["response", "utterance", "text", "comment"]:
        if key in obj:
            response = str(obj.get(key, "")).strip()
            if response:
                break

    if not response and context:
        response = context[-1]
        context = context[:-1]

    if not response:
        return None

    # Preserve speaker context as lightweight textual tags so the model can use
    # turn-taking and role cues that are often predictive of sarcasm.
    tagged_context = []
    for idx, turn_text in enumerate(context):
        spk = context_speakers[idx] if idx < len(context_speakers) and context_speakers[idx] else "UNK"
        tagged_context.append(f"[{spk}] {turn_text}")

    response_speaker = str(obj.get("speaker", "")).strip() or "UNK"
    tagged_response = f"[{response_speaker}] {response}"

    turns = tagged_context + [tagged_response]
    return turns, label


def load_samples(dataset_path):
    samples = []

    def _append_record(record):
        parsed = _extract_turns_and_label(record)
        if parsed is None:
            return
        turns, label = parsed
        samples.append((turns, label))

    # First, try parsing as regular JSON (single object, dict-of-records, or list).
    try:
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for record in data:
                if isinstance(record, dict):
                    _append_record(record)
        elif isinstance(data, dict):
            # If this is a single record with sarcasm labels, parse it directly.
            if any(k in data for k in ["headline", "context", "response", "utterance", "text", "comment"]):
                _append_record(data)
            else:
                # Otherwise treat as id -> record mapping (common in reddit datasets).
                for record in data.values():
                    if isinstance(record, dict):
                        _append_record(record)
    except json.JSONDecodeError:
        # Fallback: JSONL format (one JSON object per line).
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    _append_record(obj)

    if not samples:
        raise ValueError(f"No usable samples found in {dataset_path}")
    return samples


@dataclass
class EncodedDialogue:
    input_ids: np.ndarray
    token_attention: np.ndarray
    turn_mask: np.ndarray
    labels: np.ndarray


def build_turn_matrix(turns, max_turns):
    turns = turns[-max_turns:]
    padded = [""] * (max_turns - len(turns)) + turns
    mask = [0] * (max_turns - len(turns)) + [1] * len(turns)
    return padded, mask


def encode_samples(samples, tokenizer, max_turns, max_len):
    all_turns = []
    all_masks = []
    labels = []
    for turns, label in samples:
        padded_turns, turn_mask = build_turn_matrix(turns, max_turns)
        all_turns.extend(padded_turns)
        all_masks.append(turn_mask)
        labels.append(label)

    encoded = tokenizer(
        all_turns,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    num_samples = len(samples)
    input_ids = encoded["input_ids"].reshape(num_samples, max_turns, max_len)
    token_attention = encoded["attention_mask"].reshape(num_samples, max_turns, max_len)
    turn_mask = np.array(all_masks, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)

    return EncodedDialogue(input_ids, token_attention, turn_mask, labels)


class DialogueSarcasmDataset(Dataset):
    def __init__(self, encoded: EncodedDialogue, indices):
        self.encoded = encoded
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.tensor(self.encoded.input_ids[i], dtype=torch.long),
            torch.tensor(self.encoded.token_attention[i], dtype=torch.long),
            torch.tensor(self.encoded.turn_mask[i], dtype=torch.long),
            torch.tensor(self.encoded.labels[i], dtype=torch.long),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, num_units, num_heads=2, dropout_rate=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.k_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.v_proj = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU())
        self.output_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, queries, keys, values, key_mask=None):
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        q_ = torch.cat(torch.chunk(q, self.num_heads, dim=2), dim=0)
        k_ = torch.cat(torch.chunk(k, self.num_heads, dim=2), dim=0)
        v_ = torch.cat(torch.chunk(v, self.num_heads, dim=2), dim=0)

        scores = torch.bmm(q_, k_.permute(0, 2, 1)) / (k_.size(-1) ** 0.5)
        if key_mask is not None:
            expanded_mask = key_mask.unsqueeze(1).expand(-1, queries.size(1), -1)
            expanded_mask = expanded_mask.repeat(self.num_heads, 1, 1)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = self.output_dropout(weights)
        outputs = torch.bmm(weights, v_)
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)
        return outputs + queries


class ContextualEmotionEncoder(nn.Module):
    def __init__(self, input_dim=768, emotion_dim=64, num_heads=2, dropout=0.3):
        super().__init__()
        self.emotion_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, emotion_dim),
        )
        self.bilstm = nn.LSTM(
            emotion_dim,
            emotion_dim // 2,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
        )
        self.attn = MultiHeadAttention(emotion_dim, num_heads=num_heads, dropout_rate=dropout)
        self.label_embeddings = nn.Parameter(torch.randn(1, 2, emotion_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, turn_reprs, turn_mask):
        # turn_reprs: (batch, turns, 768)
        emotion_turns = self.emotion_proj(turn_reprs)
        emotion_turns = emotion_turns * turn_mask.unsqueeze(-1).float()
        seq_out, _ = self.bilstm(emotion_turns)
        seq_out = self.dropout(seq_out) * turn_mask.unsqueeze(-1).float()

        batch_size = seq_out.size(0)
        queries = self.label_embeddings.expand(batch_size, -1, -1)
        attn_out = self.attn(queries, seq_out, seq_out, key_mask=turn_mask)
        return attn_out.reshape(batch_size, -1)


class DialogueContextEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, num_layers=1, max_turns=4, dropout=0.2):
        super().__init__()
        self.turn_pos = nn.Embedding(max_turns, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.summary_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, turn_reprs, turn_mask):
        batch_size, turns, hidden = turn_reprs.shape
        pos_ids = torch.arange(turns, device=turn_reprs.device).unsqueeze(0).expand(batch_size, -1)
        x = turn_reprs + self.turn_pos(pos_ids)

        key_padding_mask = turn_mask == 0
        contextual = self.encoder(x, src_key_padding_mask=key_padding_mask)

        target_repr = contextual[:, -1, :]
        mask = turn_mask.unsqueeze(-1).float()
        mean_repr = (contextual * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.summary_proj(torch.cat([target_repr, mean_repr], dim=1))


class ACE21ContextualEmotion(nn.Module):
    def __init__(self, max_turns=4, emotion_dim=64, dropout=0.3, num_classes=2):
        super().__init__()
        sbert = SentenceTransformer("bert-base-nli-mean-tokens")
        self.bert = sbert._modules["0"].auto_model
        del sbert

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-4:].parameters():
            param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        self.context_encoder = DialogueContextEncoder(max_turns=max_turns)
        self.emotion_encoder = ContextualEmotionEncoder(input_dim=768, emotion_dim=emotion_dim, dropout=dropout)

        self.gate = nn.Sequential(
            nn.Linear(768 + 2 * emotion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * emotion_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 + 2 * emotion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, token_attention, turn_mask):
        # input_ids/token_attention: (batch, turns, max_len)
        batch_size, turns, max_len = input_ids.shape
        flat_ids = input_ids.view(batch_size * turns, max_len)
        flat_attn = token_attention.view(batch_size * turns, max_len)

        bert_outputs = self.bert(input_ids=flat_ids, attention_mask=flat_attn)
        token_embeddings = bert_outputs.last_hidden_state
        token_mask = flat_attn.unsqueeze(-1).float()
        pooled_flat = (token_embeddings * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)
        pooled = pooled_flat.view(batch_size, turns, -1)

        context_repr = self.context_encoder(pooled, turn_mask)
        emotion_repr = self.emotion_encoder(pooled, turn_mask)

        fusion_input = torch.cat([context_repr, emotion_repr], dim=1)
        gate = self.gate(fusion_input)
        fused = torch.cat([context_repr, gate * emotion_repr], dim=1)
        return self.classifier(fused)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for turn_ids, token_mask, turn_mask, y in loader:
            turn_ids = turn_ids.to(device)
            token_mask = token_mask.to(device)
            turn_mask = turn_mask.to(device)
            logits = model(turn_ids, token_mask, turn_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    recalls = recall_score(all_labels, all_preds, labels=[0, 1], average=None, zero_division=0)
    return acc, macro_f1, recalls, all_labels, all_preds


def build_warmup_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        remaining = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(remaining)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("ACE 2.1: Dialogue Context + Contextual Emotion Embeddings")
    print("=" * 70)
    print(f"\n[1/5] Loading dataset from: {args.dataset}")
    samples = load_samples(args.dataset)
    labels = [label for _, label in samples]
    print(
        f"  Loaded {len(samples)} samples "
        f"({sum(labels)} sarcastic, {len(labels) - sum(labels)} non-sarcastic)"
    )

    print("\n[2/5] Preparing tokenizer and encoding dialogue turns...")
    _sbert_tmp = SentenceTransformer("bert-base-nli-mean-tokens")
    tokenizer = _sbert_tmp._modules["0"].tokenizer
    del _sbert_tmp

    encoded = encode_samples(samples, tokenizer, max_turns=args.max_turns, max_len=args.max_len)
    print(f"  Encoded tensor shape: {encoded.input_ids.shape} (N, turns, max_len)")

    print("\n[3/5] Train/test split...")
    indices = list(range(len(samples)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=encoded.labels,
    )
    train_loader = DataLoader(
        DialogueSarcasmDataset(encoded, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        DialogueSarcasmDataset(encoded, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    print("\n[4/5] Building ACE 2.1 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACE21ContextualEmotion(max_turns=args.max_turns).to(device)
    print(f"  Device: {device}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  Affective branch: contextual emotion embeddings (no static lexicon counts)")

    bert_params = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "bert" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": args.bert_lr},
            {"params": other_params, "lr": args.other_lr},
        ],
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = build_warmup_scheduler(optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio)

    print("\n[5/5] Training...")
    best_acc = 0.0
    best_macro_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for turn_ids, token_mask, turn_mask, y in train_loader:
            turn_ids = turn_ids.to(device)
            token_mask = token_mask.to(device)
            turn_mask = turn_mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(turn_ids, token_mask, turn_mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(train_loader), 1)

        test_acc, macro_f1, recalls, _, _ = evaluate(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

        current_lrs = [group["lr"] for group in optimizer.param_groups]

        print(
            f"  Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
            f"Macro F1: {macro_f1:.4f}, Recall(Non-Sarc): {recalls[0]:.4f}, "
            f"Recall(Sarc): {recalls[1]:.4f}, "
            f"LRs: [{current_lrs[0]:.2e}, {current_lrs[1]:.2e}]"
            f"{'  *best acc*' if test_acc == best_acc else ''}"
            f"{'  *best f1*' if macro_f1 == best_macro_f1 else ''}"
        )

    print(f"\n{'=' * 70}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"{'=' * 70}")

    final_acc, final_macro_f1, final_recalls, all_labels, all_preds = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")
    print(f"Final Macro F1: {final_macro_f1:.4f}")
    print(f"Final Recall(Non-Sarc): {final_recalls[0]:.4f}")
    print(f"Final Recall(Sarc): {final_recalls[1]:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Not Sarcastic", "Sarcastic"]))


if __name__ == "__main__":
    main()
