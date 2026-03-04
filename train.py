import tqdm
import torch
import torch.nn as nn

from utils.dataset import build_data
from utils.args import parse_args
from utils.model import build_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_label(batch):
    return batch.get("label", batch.get("labels"))


def main():
    args = parse_args()

    data_dir   = f"{args.datapath}/data"
    labels_dir = f"{args.datapath}/label.txt"
    train_loader, val_loader, test_loader = build_data(data_dir, labels_dir, batch_size=args.batch_size, mode=args.mode)

    model = build_model(mode=args.mode, n_classes=args.num_classes)

    train_model(model, train_loader, val_loader, num_epochs=args.num_epochs, device=device, mode=args.mode)
    validate(model, test_loader, device=device, mode=args.mode)


def train_model(model, train_loader, val_loader, num_epochs, device, mode):
    criterion = nn.CrossEntropyLoss()
    if mode == "text":
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    elif mode == "image":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    elif mode == "multimodal":
        optimizer = torch.optim.AdamW([
            {"params": model.text_encoder.parameters(), "lr": 2e-5},
            {"params": model.image_encoder.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ])
    else:
        raise ValueError("mode must be 'text', 'image', or 'multimodal'")

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            if mode == "text":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask)

            elif mode == "image":
                images = batch["images"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(images)

            elif mode == "multimodal":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                images = batch["images"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               images=images)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Train Acc : {correct/total:.4f}")

        validate(model, val_loader, device, mode)


def validate(model, val_loader, device, mode="text"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            if mode == "text":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask)

            elif mode == "image":
                images = batch["images"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(images)

            elif mode == "multimodal":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                images = batch["images"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               images=images)

            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"Validation Accuracy: {correct/total:.4f}")