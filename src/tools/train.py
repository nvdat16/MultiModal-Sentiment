import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from ..dataset import build_data
from ..model import build_model
from ..utils.args import parse_args


def _get_label(batch):
    return batch.get("label", batch.get("labels"))


def _forward_batch(model, batch, mode, device):
    labels = _get_label(batch).to(device)

    if mode == "text":
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
    elif mode == "image":
        logits = model(batch["image"].to(device))
    elif mode == "multimodal":
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            images=batch["image"].to(device),
        )
    else:
        raise ValueError("mode must be 'text', 'image', or 'multimodal'")

    return logits, labels


def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_sample_batch(data_loader):
    return next(iter(data_loader))


def print_model_summary(model, mode):
    total_params, trainable_params = count_parameters(model)
    print("\nModel Summary")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)


def benchmark_inference_speed(model, sample_batch, mode, device, warmup_steps=10, run_steps=50):
    model = model.to(device)
    model.eval()

    def run_inference():
        with torch.inference_mode():
            if mode == "text":
                return model(
                    input_ids=sample_batch["input_ids"].to(device),
                    attention_mask=sample_batch["attention_mask"].to(device),
                )
            if mode == "image":
                return model(sample_batch["image"].to(device))
            if mode == "multimodal":
                return model(
                    input_ids=sample_batch["input_ids"].to(device),
                    attention_mask=sample_batch["attention_mask"].to(device),
                    images=sample_batch["image"].to(device),
                )
            raise ValueError("mode must be 'text', 'image', or 'multimodal'")

    for _ in range(warmup_steps):
        run_inference()
    _sync_device(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(run_steps):
            run_inference()
        end.record()
        torch.cuda.synchronize(device)
        total_ms = start.elapsed_time(end)
    else:
        import time

        start_time = time.perf_counter()
        for _ in range(run_steps):
            run_inference()
        total_ms = (time.perf_counter() - start_time) * 1000

    batch_size = sample_batch["image"].size(0) if mode != "text" else sample_batch["input_ids"].size(0)
    avg_ms_per_batch = total_ms / run_steps
    avg_ms_per_sample = avg_ms_per_batch / batch_size
    samples_per_second = 1000.0 / avg_ms_per_sample if avg_ms_per_sample > 0 else float("inf")

    print("\nInference Speed")
    print("=" * 60)
    print(f"Average latency: {avg_ms_per_batch:.2f} ms/batch")
    print(f"Average latency: {avg_ms_per_sample:.2f} ms/sample")
    print(f"Throughput: {samples_per_second:.2f} samples/sec")
    print("=" * 60)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def _get_multimodal_param_groups(model, base_lr, text_lr=None, image_lr=None, fusion_lr=None):
    groups = []
    used = set()

    def add_group(params, lr):
        params = [p for p in params if p.requires_grad and id(p) not in used]
        if not params:
            return
        for p in params:
            used.add(id(p))
        groups.append({"params": params, "lr": lr})

    text_params = []
    if hasattr(model, "text_encoder"):
        text_params = list(model.text_encoder.parameters())
    elif hasattr(model, "text_model"):
        text_params = list(model.text_model.parameters())
    add_group(text_params, text_lr or base_lr)

    image_params = []
    if hasattr(model, "image_encoder"):
        image_params = list(model.image_encoder.parameters())
    elif hasattr(model, "image_model"):
        image_params = list(model.image_model.parameters())
    add_group(image_params, image_lr or base_lr)

    fusion_params = []
    if hasattr(model, "fusion"):
        fusion_params = list(model.fusion.parameters())
    if hasattr(model, "classifier"):
        fusion_params += list(model.classifier.parameters())
    add_group(fusion_params, fusion_lr or base_lr)

    remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": base_lr})

    return groups


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    mode,
    lr=1e-4,
    text_lr=None,
    image_lr=None,
    fusion_lr=None,
    save_path=None,
):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.46, 0.34, 0.20], device=device))

    if mode == "multimodal":
        optimizer = torch.optim.AdamW(
            _get_multimodal_param_groups(model, lr, text_lr=text_lr, image_lr=image_lr, fusion_lr=fusion_lr)
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.to(device)

    best_acc = 0.0
    best_model_state = None
    save_path = Path(save_path) if save_path is not None else None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            logits, labels = _forward_batch(model, batch, mode, device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Train Loss: {total_loss / max(len(train_loader), 1):.4f}")

        acc, precision, recall, f1 = validate(model, val_loader, criterion, device, mode)
        if acc > best_acc:
            best_acc = acc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
            print(
                f"Saved best model (acc={best_acc:.4f}, precision={precision:.4f}, "
                f"recall={recall:.4f}, f1={f1:.4f})"
            )

        print("-" * 30)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def validate(model, val_loader, criterion, device, mode="text", split_name="Validation"):
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc=split_name):
            logits, labels = _forward_batch(model, batch, mode, device)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{split_name} Results")
    print("-" * 30)
    print(f"Loss: {total_loss / max(len(val_loader), 1):.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return acc, precision, recall, f1


def evaluate(model, test_loader, device, mode):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.46, 0.34, 0.20], device=device))
    acc, precision, recall, f1 = validate(model, test_loader, criterion, device, mode, split_name="Test")
    print(
        f"\nBest Model Test Metrics -> Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1-Score: {f1:.4f}"
    )
    return acc, precision, recall, f1


def main():
    args = parse_args()
    set_random_seed(args.random_seed)

    project_root = Path(__file__).resolve().parents[2]
    data_dir = 'dataset/MVSA/data'
    label_path = 'dataset/MVSA/labelResultAll.txt'

    run_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_loader, val_loader, test_loader = build_data(
        str(data_dir),
        str(label_path),
        batch_size=args.batch_size,
        mode=args.mode,
        text_model_name=args.text_model,
        max_length=getattr(args, "max_length", 64),
    )

    model = build_model(
        mode=args.mode,
        n_classes=args.num_classes,
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        fusion_type=args.fusion_type,
    )

    sample_batch = get_sample_batch(train_loader)
    print_model_summary(model, args.mode)

    best_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        device=run_device,
        mode=args.mode,
        lr=args.lr,
        text_lr=getattr(args, "text_lr", None),
        image_lr=getattr(args, "image_lr", None),
        fusion_lr=getattr(args, "fusion_lr", None),
        save_path=project_root / f"{args.mode}.pth",
    )

    if best_model is not None:
        print("\nEvaluating best model on test set...")
        benchmark_inference_speed(best_model, sample_batch, args.mode, run_device)
        evaluate(best_model, test_loader, run_device, args.mode)


if __name__ == "__main__":
    main()