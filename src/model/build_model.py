import torch
from .model import TextClassifier, ImageClassifier, MultiModalClassifier


def build_model(mode, n_classes, text_model_name="bert-base-uncased", image_model_name="resnet18"):
    if mode == "text":
        return TextClassifier(n_classes, model_name=text_model_name)
    elif mode == "image":
        return ImageClassifier(n_classes, model_name=image_model_name)
    elif mode == "multimodal":
        return MultiModalClassifier(
            n_classes,
            text_model_name=text_model_name,
            image_model_name=image_model_name,
        )
    else:
        raise ValueError("mode must be 'text', 'image', or 'multimodal'")


def test_text_model():
    model = TextClassifier(n_classes=3)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids, attention_mask)
    print("Text model output shape:", output.shape)


def test_image_model():
    model = ImageClassifier(n_classes=3)
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