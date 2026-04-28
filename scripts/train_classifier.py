import os
import json
import argparse
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

DEVICE = "cuda" if torch.cuda.is_available() else"cpu"
print("Using:", DEVICE)

def __init__(self, df: pd.DataFrame, tokenizer, label2id: dict, max_len: int = 256):
    self.df = df.reset_index(drop=True)
    self.tokenizer = tokenizer
    self.label2id = label2id
    self.max_len = max_len

def __len__(self):
    return len(self.df)

def __getitem__(self, idx):
    row = self.df.iloc[idx]
    text = str(row["ticket_text"])
    label_id = self.label2id[row["category"]]

    enc = self.tokenizer(
        text,
        truncation = True,
        padding = "max_length",
        max_length = self.max_len,
        return_tensors='pt',
    )
    item = {k: v.squeeze(0) for k, v in enc.items()}
    item["labels"] = torch.tensors(label_id)
    return item

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k:v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch) 
        loss  = out.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

def evaluate(model, loader, id2label):
    model.eval()
    preds = []
    labels = []
    for batch in loader:
        lbls = batch["labels"].tolist()
        batch = {k:v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        p = logits.argmax(dim=-1).detach().cpu().tolist()
        preds += p; labels +=lbls
    report = classification_report(labels, preds, target_names=[id2label[i]for i in range(len(id2label))], digits = 4)
    return report

# scripts/train_classifier.py
import os, json, argparse, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TicketDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = str(row["ticket_text"])
        label = self.label2id[row["category"]]
        enc = self.tok(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k:v.squeeze(0) for k,v in enc.items()}
        item["labels"] = torch.tensor(label)
        return item

def train_epoch(model, loader, optimizer, scheduler):
    model.train(); total=0.0
    for batch in loader:
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        out = model(**batch); loss = out.loss
        loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total += loss.item()
    return total / max(1, len(loader))

@torch.inference_mode()
def evaluate(model, loader, id2label):
    model.eval()
    preds = []
    labels = []

    for batch in loader:
        lbls = batch["labels"].tolist()
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        logits = model(**batch).logits
        p = logits.argmax(dim=-1).detach().cpu().tolist()
        preds += p; labels += lbls
    report = classification_report(labels, preds, target_names=[id2label[i] for i in range(len(id2label))], digits=4)
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tickets/tickets_synthetic.csv")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out_dir", default="models/checkpoints/distilbert-helpdesk")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.data).dropna()
    labels = sorted(df["category"].unique().tolist())
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    json.dump({"label2id":label2id,"id2label":id2label}, open(os.path.join(args.out_dir,"labels.json"),"w"))

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = TicketDataset(df, tok, label2id)
    val_len = max(60, int(0.15*len(ds))); train_len = len(ds)-val_len
    tr, va = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True); val = DataLoader(va, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels), id2label=id2label, label2id=label2id).to(DEVICE)
    opt = AdamW(model.parameters(), lr=args.lr)
    steps = len(trl)*max(1,args.epochs)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1*steps), num_training_steps=steps)

    for e in range(args.epochs):
        loss = train_epoch(model, trl, opt, sch)
        print(f"Epoch {e+1}/{args.epochs} - train loss: {loss:.4f}")
        print(evaluate(model, val, id2label))

    model.save_pretrained(args.out_dir); tok.save_pretrained(args.out_dir)
    print("✅ Saved to", args.out_dir)

if __name__ == "__main__":
    main()









