import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models


TEXT_MODEL_ALIASES = {
    "bert": "bert-base-uncased",
    "robert": "roberta-base",
    "roberta": "roberta-base",
}

IMAGE_MODEL_ALIASES = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "mobilenet_v1": "mobilenet_v1",
    "mobilenet_v2": "mobilenet_v2",
    "efficientnet_b0": "efficientnet_b0",
}


class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.3):
        super().__init__()
        model_name = TEXT_MODEL_ALIASES.get(model_name, model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.dropout(cls)


class ImageEncoder(nn.Module):
    def __init__(self, model_name="resnet18"):
        super().__init__()
        model_name = IMAGE_MODEL_ALIASES.get(model_name, model_name)
        self.model_name = model_name

        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            self.output_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(pretrained=True)
            self.output_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.output_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(
                "image_model must be one of: 'resnet18', 'mobilenetv1'/'mobilenet_v2', or 'efficientnet_b0'"
            )

    def forward(self, images):
        return self.backbone(images)


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


# Backward-compatible aliases
BERTEncoder = TextEncoder
ResNetEncoder = ImageEncoder
BERTClassifier = TextClassifier
ResNetClassifier = ImageClassifier