"""CNN plus non-causal linear attention, using the team's shared blocks."""
import torch.nn as nn

from models.common import CNNBackbone, ClassificationHead, EncoderBlock, SinusoidalPositionalEncoding
from models.linear_attention import LinearSelfAttention


class HybridCNNLinearAttention(nn.Module):
    def __init__(self, in_features=40, d_model=64, cnn_kernel_size=3, cnn_layers=2,
                 n_heads=4, n_attn_layers=1, ffn_mult=2, num_classes=3, dropout=0.1,
                 use_pos_encoding=True, **_unused):
        super().__init__()
        self.backbone = CNNBackbone(in_features, d_model, cnn_kernel_size, cnn_layers, dropout)
        self.pos = SinusoidalPositionalEncoding(d_model) if use_pos_encoding else nn.Identity()
        self.encoder = nn.Sequential(*[
            EncoderBlock(LinearSelfAttention(d_model, n_heads), d_model, ffn_mult, dropout)
            for _ in range(n_attn_layers)
        ])
        self.head = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, x):
        return self.head(self.encoder(self.pos(self.backbone(x))))
