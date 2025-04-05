import torch
import torch.nn as nn

class PoseTransformer(nn.Module):
    def __init__(self, input_dim=99, model_dim=128, num_classes=3, seq_len=30):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):  # x: [batch, seq_len, input_dim]
        x = self.embedding(x)  # [batch, seq_len, model_dim]
        x = self.transformer(x)  # [batch, seq_len, model_dim]
        x = x.mean(dim=1)  # [batch, model_dim]
        return self.classifier(x)
