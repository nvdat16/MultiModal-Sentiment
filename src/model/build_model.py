import torch
from .model import BERTClassifier, ResNetClassifier, MultiModalClassifier, CrossAttentionMultiModalClassifier


def build_model(mode, n_classes, *args, **kwargs):
    text_model_name = kwargs.get("text_model_name", kwargs.get("text_model", "bert-base-uncased"))
    image_model_name = kwargs.get("image_model_name", kwargs.get("image_model", "resnet18"))
    fusion_type = kwargs.get("fusion_type", "concat")

    if mode == "text":
        return BERTClassifier(n_classes, text_model_name)
    elif mode == "image":
        return ResNetClassifier(n_classes, image_model_name)
    elif mode == "multimodal":
        if fusion_type == "cross_attention":
            return CrossAttentionMultiModalClassifier(
                n_classes,
                text_model_name=text_model_name,
                image_model_name=image_model_name,
            )
        return MultiModalClassifier(
            n_classes,
            text_model_name=text_model_name,
            image_model_name=image_model_name,
        )
    else:
        raise ValueError("mode must be 'text', 'image', or 'multimodal'")

