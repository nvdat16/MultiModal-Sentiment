import os
import re
import pandas as pd
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer


BERT_MODEL   = "bert-base-uncased"
MAX_LENGTH   = 128
BATCH_SIZE   = 32


# Label processing
def load_labels(labels_dir: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_dir, sep="\t")

    def get_majority_multimodal(row):
        raw_labels = [row[col] for col in row.index if "text,image" in col]
        texts, images = [], []
        for item in raw_labels:
            t, i = item.split(",")
            texts.append(t)
            images.append(i)

        def get_winner(val_list):
            counts = Counter(val_list)
            for val, count in counts.items():
                if count > 1:
                    return val
            return None

        final_text  = get_winner(texts)
        final_image = get_winner(images)
        return f"{final_text},{final_image}" if final_text and final_image else "invalid"

    labels["majority_label"] = labels.apply(get_majority_multimodal, axis=1)
    labels[["text_label", "image_label"]] = labels["majority_label"].str.split(",", expand=True)

    labels["label"] = labels.apply(
        lambda row: row["text_label"] if row["text_label"] == row["image_label"] else "invalid",
        axis=1,
    )

    # Loại invalid
    labels = labels[labels["label"] != "invalid"].reset_index(drop=True)
    return labels[["ID", "label"]]


# Text processing
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)   # bỏ URL
    text = re.sub(r"<.*?>", "", text)     # bỏ HTML tag
    text = re.sub(r"\s+", " ", text)      # normalize khoảng trắng
    return text.strip()


def load_texts(data_dir: str) -> pd.DataFrame:
    records = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            path = os.path.join(data_dir, file)
            id_  = int(file.split(".")[0])
            with open(path, "r", encoding="utf-8") as f:
                content = clean_text(f.read().strip())
            records.append((id_, content))
    return pd.DataFrame(records, columns=["ID", "text"])


# Image processing
def load_image_paths(data_dir: str) -> pd.DataFrame:
    records = []
    for file in os.listdir(data_dir):
        if file.endswith((".jpg", ".png", ".jpeg")):
            id_  = int(file.split(".")[0])
            path = os.path.join(data_dir, file)
            records.append((id_, path))
    return pd.DataFrame(records, columns=["ID", "image_path"])


# Kết hợp tất cả vào 1 DataFrame
def build_dataframe(data_dir: str, labels_dir: str) -> pd.DataFrame:
    df_labels = load_labels(labels_dir)
    df_text   = load_texts(data_dir)
    df_image  = load_image_paths(data_dir)

    df = df_labels.merge(df_text,  on="ID", how="left")
    df = df.merge(df_image, on="ID", how="left")

    # Loại bỏ row thiếu text hoặc image
    df = df.dropna(subset=["text", "image_path"]).reset_index(drop=True)
    return df


class Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer=None, max_length: int = MAX_LENGTH, mode: str = "multimodal"):
        assert mode in ("multimodal", "text", "image")
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.mode       = mode
        self.transform  = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.label_map   = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = self.label_map[row["label"]]
        result = {"label": label}

        if self.mode in ("multimodal", "text"):
            encoding = self.tokenizer(
                row["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"]      = encoding["input_ids"].squeeze(0)
            result["attention_mask"] = encoding["attention_mask"].squeeze(0)

        if self.mode in ("multimodal", "image"):
            image = Image.open(row["image_path"]).convert("RGB")
            result["image"] = self.transform(image)

        return result


# Get dataLoader
def get_dataloader(df, tokenizer=None, batch_size=BATCH_SIZE, shuffle=True, mode="multimodal"):
    dataset = Dataset(df, tokenizer, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


# Data pipeline
def build_data(data_dir, labels_dir, batch_size=BATCH_SIZE, mode="multimodal"):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    df = build_dataframe(data_dir, labels_dir)
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())

    # Train / val / test split 80 / 10 / 10
    train_df = df.sample(frac=0.8, random_state=42)
    temp_df  = df.drop(train_df.index)
    val_df   = temp_df.sample(frac=0.5, random_state=42)
    test_df  = temp_df.drop(val_df.index)

    train_loader = get_dataloader(train_df, tokenizer, batch_size, shuffle=True, mode=mode)
    val_loader   = get_dataloader(val_df,   tokenizer, batch_size, shuffle=False, mode=mode)
    test_loader  = get_dataloader(test_df,  tokenizer, batch_size, shuffle=False, mode=mode)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_dir   = "dataset/data"
    labels_dir = "dataset/label.txt"
    train_loader, val_loader, test_loader = build_data(data_dir, labels_dir, batch_size=16, mode="multimodal")
