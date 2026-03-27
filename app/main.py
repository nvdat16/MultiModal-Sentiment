from fastapi import FastAPI, File, UploadFile, Form, Query

import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model


app = FastAPI()

# load model
models = {}

def get_model(mode: str):
    if mode not in models:
        m = build_model(mode=mode, n_classes=3)
        m.load_state_dict(torch.load(f"checkpoints/{mode}.pth", map_location="cpu"))
        m.eval()
        models[mode] = m
    return models[mode]

labels = ["negative", "neutral", "positive"]

@app.get("/")
async def root():
    return {"Status": "OK"}

@app.post("/predict")
async def predict(
    mode: str = Query(default="multimodal", enum=["multimodal", "text", "image"]),
    file: UploadFile = File(None),
    text: str = Form(None)
):
    model = get_model(mode)
    # preprocess
    input_ids, attention_mask = ''
    image_tensor = ''

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image_tensor
        )

        pred = torch.argmax(outputs, dim=1).item()

    return {
        "text": text,
        "sentiment": labels[pred]
    }