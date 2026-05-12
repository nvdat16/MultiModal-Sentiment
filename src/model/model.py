import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models


TEXT_MODEL_ALIASES = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}

IMAGE_MODEL_ALIASES = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "mobilenet_v1": "mobilenet_v2",
    "mobilenet_v2": "mobilenet_v2",
    "efficientnet_b0": "efficientnet_b0",
}


def _load_backbone_with_weights(name, builder, weights_attr):
    try:
        weights = getattr(models, weights_attr).DEFAULT
        return builder(weights=weights)
    except Exception:
        return builder(pretrained=True)


class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.3):
        super().__init__()
        model_name = TEXT_MODEL_ALIASES.get(model_name, model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, return_sequence=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence = self.dropout(outputs.last_hidden_state)
        if return_sequence:
            return sequence
        return sequence[:, 0, :]


class ImageEncoder(nn.Module):
    def __init__(self, model_name="resnet18", dropout=0.3):
        super().__init__()
        model_name = IMAGE_MODEL_ALIASES.get(model_name, model_name)
        self.model_name = model_name
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if model_name == "resnet18":
            backbone = _load_backbone_with_weights("resnet18", models.resnet18, "ResNet18_Weights")
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
            self.output_dim = backbone.fc.in_features
        elif model_name == "resnet34":
            backbone = _load_backbone_with_weights("resnet34", models.resnet34, "ResNet34_Weights")
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
            self.output_dim = backbone.fc.in_features
        elif model_name == "mobilenet_v2":
            backbone = _load_backbone_with_weights("mobilenet_v2", models.mobilenet_v2, "MobileNet_V2_Weights")
            self.feature_extractor = backbone.features
            self.output_dim = backbone.classifier[1].in_features
        elif model_name == "efficientnet_b0":
            backbone = _load_backbone_with_weights("efficientnet_b0", models.efficientnet_b0, "EfficientNet_B0_Weights")
            self.feature_extractor = backbone.features
            self.output_dim = backbone.classifier[1].in_features
        else:
            raise ValueError(
                "image_model must be one of: 'resnet18', 'resnet34', 'mobilenet_v1'/'mobilenet_v2', or 'efficientnet_b0'"
            )

    def forward(self, images, return_sequence=False):
        feature_map = self.feature_extractor(images)
        if return_sequence:
            tokens = feature_map.flatten(2).transpose(1, 2)
            return self.dropout(tokens)
        pooled = self.pool(feature_map).flatten(1)
        return self.dropout(pooled)


class TextClassifier(nn.Module):
    def __init__(self, n_classes, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = TextEncoder(model_name=model_name)
        self.fc = nn.Linear(self.encoder.output_dim, n_classes)

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)
        return self.fc(features)


class ImageClassifier(nn.Module):
    def __init__(self, n_classes, model_name="resnet18"):
        super().__init__()
        self.encoder = ImageEncoder(model_name=model_name)
        self.fc = nn.Linear(self.encoder.output_dim, n_classes)

    def forward(self, images):
        features = self.encoder(images)
        return self.fc(features)


class MultiModalClassifier(nn.Module):
    def __init__(self, n_classes, text_model_name="bert-base-uncased", image_model_name="resnet18"):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=text_model_name)
        self.image_encoder = ImageEncoder(model_name=image_model_name)

        fused_dim = self.text_encoder.output_dim + self.image_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        text_feat = self.text_encoder(input_ids, attention_mask)
        image_feat = self.image_encoder(images)
        fused = torch.cat([text_feat, image_feat], dim=1)
        return self.classifier(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim, image_dim, fusion_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_to_image = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = fusion_dim * 4

    def forward(self, text_tokens, image_tokens):
        text_tokens = self.text_proj(text_tokens)
        image_tokens = self.image_proj(image_tokens)

        text_context, _ = self.text_to_image(query=text_tokens, key=image_tokens, value=image_tokens, need_weights=False)
        image_context, _ = self.image_to_text(query=image_tokens, key=text_tokens, value=text_tokens, need_weights=False)

        text_cls = text_context[:, 0, :]
        image_pooled = image_context.mean(dim=1)
        text_pooled = text_tokens[:, 0, :]
        image_cls = image_tokens.mean(dim=1)

        fused = torch.cat([text_cls, image_pooled, text_pooled, image_cls], dim=1)
        return self.dropout(fused)


class CrossAttentionMultiModalClassifier(nn.Module):
    def __init__(self, n_classes, text_model_name="bert-base-uncased", image_model_name="resnet18", fusion_dim=256, num_heads=4):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=text_model_name)
        self.image_encoder = ImageEncoder(model_name=image_model_name)
        self.fusion = CrossAttentionFusion(
            text_dim=self.text_encoder.output_dim,
            image_dim=self.image_encoder.output_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        text_tokens = self.text_encoder(input_ids, attention_mask, return_sequence=True)
        image_tokens = self.image_encoder(images, return_sequence=True)
        fused = self.fusion(text_tokens, image_tokens)
        return self.classifier(fused)


# Backward-compatible aliases
BERTEncoder = TextEncoder
ResNetEncoder = ImageEncoder
BERTClassifier = TextClassifier
ResNetClassifier = ImageClassifier