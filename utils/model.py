import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.models as models



class BERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.fc(x)
        return logits
    

class ResNetClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)
        
    def forward(self, images):
        return self.resnet(images)
    

class MultiModalClassifier(nn.Module):
    def __init__(self, n_classes):
        super(MultiModalClassifier, self).__init__()
        self.text_model = BERTClassifier(n_classes)
        self.image_model = ResNetClassifier(n_classes)
        self.fc = nn.Linear(n_classes * 2, n_classes)
        
    def forward(self, input_ids, attention_mask, images):
        text_logits = self.text_model(input_ids, attention_mask)
        image_logits = self.image_model(images)
        combined = torch.cat((text_logits, image_logits), dim=1)
        output = self.fc(combined)
        return output
    

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