import os
import re
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer


DEFAULT_TEXT_MODEL = "bert-base-uncased"
MAX_LENGTH = 32


# Text processing 
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Label + Text
def load_labels_excel(path):
    df = pd.read_excel(path, sheet_name="final label")
    df.columns = [c.lower().strip() for c in df.columns]

    df["id"] = df["file name"].str.extract(r"(\d+)").astype(int)
    df["text"] = df["caption"].astype(str).apply(clean_text)
    df["label"] = df["label"].str.lower().str.strip()

    return df[["id", "text", "label"]]


# Image processing 
def load_images_from_folder(image_root):
    records = []

    for label in os.listdir(image_root):
        folder = os.path.join(image_root, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                id_ = int(os.path.splitext(file)[0])
                records.append({
                    "id": id_,
                    "image_path": os.path.join(folder, file)
                })

    return pd.DataFrame(records)


def build_dataframe(image_root, label_path):
    df_img = load_images_from_folder(image_root)
    df_lbl = load_labels_excel(label_path)

    df = df_img.merge(df_lbl, on="id")
    return df


class Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH, mode="multimodal"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = self.label_map[row["label"]]

        # TEXT fields
        if self.mode in ("text", "multimodal"):
            enc = self.tokenizer(
                row["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        # IMAGE fields
        if self.mode in ("image", "multimodal"):
            image = Image.open(row["image_path"]).convert("RGB")
            image = self.transform(image)

        # Trả về đúng fields theo mode
        if self.mode == "text":
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": label
            }
        elif self.mode == "image":
            return {
                "image": image,
                "label": label
            }
        else:  # multimodal
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "image": image,
                "label": label
            }


def get_dataloader(df, tokenizer, batch_size=16, shuffle=True, mode="multimodal"):
    dataset = Dataset(df, tokenizer, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def build_data(image_root, label_path, batch_size=16, mode="multimodal", text_model_name=DEFAULT_TEXT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    df = build_dataframe(image_root, label_path)

    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())

    train_df = df.sample(frac=0.8, random_state=42)
    val_df   = df.drop(train_df.index)
    
    test_df = val_df.sample(frac=0.5, random_state=42)
    val_df = val_df.drop(test_df.index)

    train_loader = get_dataloader(train_df, tokenizer, batch_size, shuffle=True,  mode=mode)
    val_loader   = get_dataloader(val_df,   tokenizer, batch_size, shuffle=False, mode=mode)
    test_loader  = get_dataloader(test_df,  tokenizer, batch_size, shuffle=False, mode=mode)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    image_root = "dataset/Images/Images"
    label_path = "dataset/LabeledText.xlsx"

    train_loader, val_loader, test_loader = build_data(
        image_root,
        label_path,
        batch_size=16,
        mode='text',
        text_model_name='roberta-base'
    )
    print('Load dataset: Done')