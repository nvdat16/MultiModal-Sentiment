from .model import TextClassifier, ImageClassifier, MultiModalClassifier, CrossAttentionMultiModalClassifier


MODEL_ALIAS = {
    "bert": "bert-base-uncased",
    "bert-base": "bert-base-uncased",
    "bert-base-uncased": "bert-base-uncased",
    "roberta": "roberta-base",
    "mobilenetv2": "mobilenet_v2",
    "mobilenet_v2": "mobilenet_v2",
    "mobilenetv1": "mobilenet_v2",
    "mobilenet_v1": "mobilenet_v2",
}


def _normalize_model_name(name):
    return MODEL_ALIAS.get(name, name)


def build_model(mode, n_classes, *args, **kwargs):
    text_model_name = _normalize_model_name(kwargs.get("text_model_name", kwargs.get("text_model", "bert-base-uncased")))
    image_model_name = _normalize_model_name(kwargs.get("image_model_name", kwargs.get("image_model", "resnet18")))
    fusion_type = kwargs.get("fusion_type", "concat")

    if mode == "text":
        return TextClassifier(n_classes, text_model_name)
    elif mode == "image":
        return ImageClassifier(n_classes, image_model_name)
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

