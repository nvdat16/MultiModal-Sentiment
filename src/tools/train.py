import random
import numpy as np
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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_random_seed(args.random_seed)

    image_root = "dataset/Images/Images"
    label_path = "dataset/LabeledText.xlsx"

    train_loader, val_loader, test_loader = build_data(
        image_root,
        label_path,
        batch_size=args.batch_size,
        mode=args.mode,
        text_model_name=args.text_model,
    )

    model = build_model(
        mode=args.mode,
        n_classes=args.num_classes,
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        fusion_type=args.fusion_type,
    )

    best_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        device=device,
        mode=args.mode,
    )

    if best_model is not None:
        print("\nEvaluating best model on test set...")
        evaluate(best_model, test_loader, device, args.mode)


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
            {"params": model.classifier.parameters(), "lr": 1e-5},
        ])
    else:
        raise ValueError("mode must be 'text', 'image', or 'multimodal'")

    model.to(device)

    best_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            if mode == "text":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = _get_label(batch).to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

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
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Train Loss: {total_loss/len(train_loader):.4f}")

        acc, precision, recall, f1 = validate(model, val_loader, criterion, device, mode)

        if acc > best_acc:
            best_acc = acc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), f"best_model_{mode}.pth")
            print(f"Saved best model (acc={best_acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f})")

        print('-'*30)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def validate(model, val_loader, criterion, device, mode="text", split_name="Validation"):
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc=split_name):
            if mode == "text":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = _get_label(batch).to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)

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
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{split_name} Results")
    print("-" * 30)
    print(f"Loss: {total_loss/len(val_loader):.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return acc, precision, recall, f1


def evaluate(model, test_loader, device, mode):
    criterion = FocalLoss(gamma=2)
    acc, precision, recall, f1 = validate(model, test_loader, criterion, device, mode, split_name="Test")
    print(f"\nBest Model Test Metrics -> Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return acc, precision, recall, f1

if __name__ == "__main__":
    main()