import torch
import torch.nn as nn
import timm

class ViTKinshipModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, n_transformer_layers=2, nhead=8, hidden_dim=768):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.vit.head = nn.Identity()

        # Custom transformer encoder for kinship feature fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.vit.num_features * 2,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.vit.num_features * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, img1, img2):
        feat1 = self.vit(img1)
        feat2 = self.vit(img2)
        combined = torch.cat([feat1, feat2], dim=1)  # (batch, 2*features)
        # Add sequence dimension for transformer: (batch, seq_len=1, features)
        combined_seq = combined.unsqueeze(1)
        fused = self.transformer_encoder(combined_seq)
        fused = fused.squeeze(1)  # (batch, features)
        out = self.classifier(fused)
        return out.squeeze(1)
