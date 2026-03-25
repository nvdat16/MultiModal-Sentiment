from fastapi import FastAPI, File, UploadFile, Form

import torch
from src.model import build_model


app = FastAPI()

# load model
model = build_model(mode="multimodal", n_classes=3)
model.load_state_dict(torch.load("weights/model.pth", map_location="cpu"))
model.eval()

labels = ["negative", "neutral", "positive"]

@app.get("/")
async def root():
    return {"Status": "OK"}

@app.post("/predict")
async def predict(
    text: str = Form(...),
    image: UploadFile = File(...)
):
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