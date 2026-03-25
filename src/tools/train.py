import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dataset import build_data
from ..utils.args import parse_args
from ..model import build_model

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_label(batch):
    return batch.get("label", batch.get("labels"))


def main():
    args = parse_args()

    data_dir   = f"{args.datapath}/data"
    labels_dir = f"{args.datapath}/label.txt"
    train_loader, val_loader = build_data(data_dir, labels_dir, batch_size=args.batch_size, mode=args.mode)

    model = build_model(mode=args.mode, n_classes=args.num_classes)

    train_model(model, train_loader, val_loader, num_epochs=args.num_epochs, device=device, mode=args.mode)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()
    

def train_model(model, train_loader, val_loader, num_epochs, device, mode):
    criterion = FocalLoss(gamma=2)
    if mode == "text":
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    elif mode == "image":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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

        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            if mode == "text":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask)

            elif mode == "image":
                images = batch["image"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(images)

            elif mode == "multimodal":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                images = batch["image"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               images=images)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Train Loss: {total_loss/len(train_loader):.4f}")

        validate(model, val_loader, criterion, device, mode)
        print('-'*30)


def validate(model, val_loader, criterion, device, mode="text"):
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Validating"):
            if mode == "text":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = _get_label(batch).to(device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            elif mode == "image":
                images = batch["image"].to(device)
                labels = _get_label(batch).to(device)

                logits = model(images)

            elif mode == "multimodal":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                images = batch["image"].to(device)
                labels = _get_label(batch).to(device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images
                )
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)

    print("\nValidation Results")
    print("-" * 30)
    print(f"Validation Loss: {total_loss/len(val_loader):.4f}")
    print(f"Accuracy: {acc:.4f}\n")
    print("F1-Score:", f1_score(y_true, y_pred))

    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()