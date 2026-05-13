import os
import re
import unicodedata
from collections import Counter

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from transformers import AutoTokenizer


DEFAULT_TEXT_MODEL = "bert-base-uncased"
MAX_LENGTH = 64
IMAGE_SIZE = 224
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^\w\s.,!?;:'\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_texts_from_folder(data_dir, preprocess=True):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    records = []
    for file in sorted(os.listdir(data_dir)):
        if not file.lower().endswith(".txt"):
            continue

        txt_path = os.path.join(data_dir, file)
        file_id = os.path.splitext(file)[0]
        if not file_id.isdigit():
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        text = clean_text(content) if preprocess else content
        if text:
            records.append({"id": int(file_id), "text": text})

    if not records:
        raise ValueError(f"No text files found under: {data_dir}")

    return pd.DataFrame(records)


def load_images_from_folder(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    records = []
    for file in sorted(os.listdir(data_dir)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        file_id = os.path.splitext(file)[0]
        if not file_id.isdigit():
            continue

        records.append({"id": int(file_id), "image_path": os.path.join(data_dir, file)})

    if not records:
        raise ValueError(f"No image files found under: {data_dir}")

    return pd.DataFrame(records)


def _get_majority(values):
    counts = Counter(v for v in values if pd.notna(v) and str(v).strip())
    if not counts:
        return None

    value, count = counts.most_common(1)[0]
    return value if count > 1 else None


def load_labels_txt(label_path):
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    df = pd.read_csv(label_path, sep="\t")
    df.columns = [c.lower().strip() for c in df.columns]

    id_col = None
    for candidate in ["id", "file name", "filename", "file_name"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        raise ValueError("Missing ID column in label file.")

    if id_col != "id":
        df["id"] = df[id_col].astype(str).str.extract(r"(\d+)")[0]
    else:
        df["id"] = df["id"].astype(str).str.extract(r"(\d+)")[0]

    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    multimodal_cols = [c for c in df.columns if "text,image" in c]
    if not multimodal_cols:
        raise ValueError("No multimodal label columns found in label file.")

    def get_majority_multimodal(row):
        raw_labels = [row[col] for col in multimodal_cols]

        texts = []
        images = []
        for item in raw_labels:
            if pd.isna(item):
                continue
            item = str(item).strip()
            if "," not in item:
                continue
            t, i = item.split(",", 1)
            texts.append(t.strip().lower())
            images.append(i.strip().lower())

        final_text = _get_majority(texts)
        final_image = _get_majority(images)

        if final_text and final_image and final_text == final_image:
            return final_text
        return "invalid"

    df["label"] = df.apply(get_majority_multimodal, axis=1)
    df = df[df["label"].isin(LABEL_MAP)].copy()

    return df[["id", "label"]].drop_duplicates(subset=["id"], keep="first")


def build_dataframe(data_dir, label_path, preprocess=True):
    df_text = load_texts_from_folder(data_dir, preprocess=preprocess)
    df_img = load_images_from_folder(data_dir)
    df_lbl = load_labels_txt(label_path)

    df = df_text.merge(df_img, on="id", how="inner")
    df = df.merge(df_lbl, on="id", how="inner")

    if df.empty:
        raise ValueError("No matched samples after merging text, images, and labels.")

    return df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)


class SentimentDataset(TorchDataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH, mode="multimodal", train=False, preprocess=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.train = train
        self.preprocess = preprocess

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.raw_image_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def _image_transform(self):
        return self.image_transform if self.preprocess else self.raw_image_transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_name = str(row["label"]).lower().strip()
        if label_name not in LABEL_MAP:
            raise ValueError(f"Unsupported label: {row['label']}")
        label = LABEL_MAP[label_name]

        enc = None
        image = None

        if self.mode in ("text", "multimodal"):
            enc = self.tokenizer(
                clean_text(row["text"]),
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        if self.mode in ("image", "multimodal"):
            try:
                image = Image.open(row["image_path"]).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Failed to load image: {row['image_path']}") from exc
            image = self._image_transform()(image)

        if self.mode == "text":
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": label,
            }
        if self.mode == "image":
            return {"image": image, "label": label}

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": image,
            "label": label,
        }


def get_dataloader(
    df,
    tokenizer,
    batch_size=16,
    shuffle=True,
    mode="multimodal",
    max_length=MAX_LENGTH,
    train=False,
    preprocess=True,
):
    dataset = SentimentDataset(
        df,
        tokenizer,
        max_length=max_length,
        mode=mode,
        train=train,
        preprocess=preprocess,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=False,
    )


def _can_stratify(series):
    counts = series.value_counts()
    return len(counts) > 1 and counts.min() >= 2


def build_data(
    data_dir,
    label_path,
    batch_size=16,
    mode="multimodal",
    text_model_name=DEFAULT_TEXT_MODEL,
    max_length=MAX_LENGTH,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    preprocess=True,
):
    if mode not in {"text", "image", "multimodal"}:
        raise ValueError("mode must be 'text', 'image', or 'multimodal'")

    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    df = build_dataframe(data_dir, label_path, preprocess=preprocess)

    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())

    temp_size = val_size + test_size
    if temp_size <= 0 or temp_size >= 1:
        raise ValueError("val_size + test_size must be between 0 and 1")

    stratify = df["label"] if _can_stratify(df["label"]) else None
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    temp_stratify = temp_df["label"] if _can_stratify(temp_df["label"]) else None
    relative_test_size = test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True,
        stratify=temp_stratify,
    )

    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_loader = get_dataloader(
        train_df,
        tokenizer,
        batch_size,
        shuffle=True,
        mode=mode,
        max_length=max_length,
        train=True,
        preprocess=preprocess,
    )
    val_loader = get_dataloader(
        val_df,
        tokenizer,
        batch_size,
        shuffle=False,
        mode=mode,
        max_length=max_length,
        train=False,
        preprocess=preprocess,
    )
    test_loader = get_dataloader(
        test_df,
        tokenizer,
        batch_size,
        shuffle=False,
        mode=mode,
        max_length=max_length,
        train=False,
        preprocess=preprocess,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_dir = "dataset/MVSA/data"
    label_path = "dataset/MVSA/labelResultAll.txt"

    train_loader, val_loader, test_loader = build_data(
        data_dir,
        label_path,
        batch_size=16,
        mode="text",
        text_model_name="roberta-base",
        preprocess=True,
    )
    print("Load dataset: Done")