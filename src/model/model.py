import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models


class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.dropout(cls)   # shape: (B, 768)
    

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()

    def forward(self, images):
        features = self.resnet(images)
        return features  # shape: (B, 512)
    

class BERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = BERTEncoder()
        self.fc = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)  # (B, 768)
        logits = self.fc(features)
        return logits
    

class ResNetClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.fc = nn.Linear(512, n_classes)

    def forward(self, images):
        features = self.encoder(images)  # (B, 512)
        logits = self.fc(features)
        return logits
    

class MultiModalClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.text_encoder = BERTEncoder()      # 768
        self.image_encoder = ResNetEncoder()   # 512
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        text_feat = self.text_encoder(input_ids, attention_mask)
        image_feat = self.image_encoder(images)
        
        fused = torch.cat([text_feat, image_feat], dim=1)
        output = self.classifier(fused)
        
        return output