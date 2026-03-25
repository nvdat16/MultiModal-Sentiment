import torch
from .model import BERTClassifier, ResNetClassifier, MultiModalClassifier


def build_model(mode, n_classes):
    if mode == "text":
        return BERTClassifier(n_classes)
    elif mode == "image":
        return ResNetClassifier(n_classes)
    elif mode == "multimodal":
        return MultiModalClassifier(n_classes)
    else:
        raise ValueError("mode must be 'text', 'image', or 'multimodal'")

def test_text_model():
    model = BERTClassifier(n_classes=3)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids, attention_mask)
    print("Text model output shape:", output.shape)


def test_image_model():
    model = ResNetClassifier(n_classes=3)
    images = torch.randn(2, 3, 224, 224)
    output = model(images)
    print("Image model output shape:", output.shape)


def test_multimodal_model():
    model = MultiModalClassifier(n_classes=3)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones_like(input_ids)
    images = torch.randn(2, 3, 224, 224) 
    output = model(input_ids, attention_mask, images)
    print("Multimodal model output shape:", output.shape)