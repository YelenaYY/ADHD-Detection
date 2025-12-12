"""
SpatioTemporal model for ADHD fMRI:

1) TemporalEncoder: 1D CNN over time for each ROI.
2) ROITransformer: Transformer encoder over ROI tokens + positional encoding.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Expects input of shape (batch_size, seq_len, d_model) and
    adds a fixed positional encoding to the last dimension.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model

        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # Register as buffer so it moves with .to(device) but isn't a parameter
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding up to sequence length
        return x + self.pe[:, :seq_len, :]


class TemporalEncoder(nn.Module):
    """
    Temporal encoder that processes fMRI time series per ROI.

    Input:  x of shape (batch_size, T, R)
    Output: roi_embeddings of shape (batch_size, R, d_model)
    """

    def __init__(
        self,
        n_rois: int = 90,
        d_model: int = 128,
        hidden_channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.d_model = d_model

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv3 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.activation = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, T, R)
        returns: (batch_size, R, d_model)
        """
        B, T, R = x.shape
        if R != self.n_rois:
            raise ValueError(f"Expected {self.n_rois} ROIs, got {R}")

        # (B, T, R) -> (B, R, T)
        x = x.permute(0, 2, 1)
        # Flatten ROI dimension into batch: (B*R, 1, T)
        x = x.reshape(B * R, 1, T)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))  # (B*R, d_model, T)

        # Global average pooling over time
        x = x.mean(dim=-1)                   # (B*R, d_model)

        # (B*R, d_model) -> (B, R, d_model)
        x = x.view(B, R, self.d_model)
        return x


class ROITransformer(nn.Module):
    """
    Transformer over ROI tokens to capture spatial / connectivity structure.

    Input:  roi_embeddings of shape (batch_size, R, d_model)
    Output: logits of shape (batch_size, num_classes)
    """

    def __init__(
        self,
        n_rois: int = 90,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 1,  # 1 for binary (BCEWithLogitsLoss)
    ):
        super().__init__()

        self.n_rois = n_rois
        self.d_model = d_model

        # Learned embedding for ROI identity (acts like "spatial position")
        self.roi_embedding = nn.Embedding(n_rois, d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding over the sequence (CLS + ROIs)
        # Sequence length = 1 (CLS) + n_rois
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_rois + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.roi_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, roi_embeddings: torch.Tensor) -> torch.Tensor:
        """
        roi_embeddings: (batch_size, R, d_model)
        returns: logits: (batch_size, num_classes)
        """
        B, R, D = roi_embeddings.shape
        if R != self.n_rois:
            raise ValueError(f"Expected {self.n_rois} ROIs, got {R}")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")

        device = roi_embeddings.device

        # ROI identity embeddings (like positional embeddings over ROI index)
        roi_ids = torch.arange(self.n_rois, device=device).unsqueeze(0).expand(B, -1)
        roi_id_embed = self.roi_embedding(roi_ids)  # (B, R, d_model)

        # Add ROI identity embedding
        x = roi_embeddings + roi_id_embed  # (B, R, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, 1, D)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)        # (B, 1+R, d_model)

        # Add sinusoidal positional encoding over sequence dimension
        x = self.pos_encoder(x)                     # (B, 1+R, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)              # (B, 1+R, d_model)

        # CLS output
        cls_out = x[:, 0, :]                         # (B, d_model)

        logits = self.classifier(cls_out)            # (B, num_classes)
        return logits


class SpatioTemporalADHDModel(nn.Module):
    """
    Full 2-stage model:
      1) TemporalEncoder over time dimension per ROI
      2) ROITransformer over ROI tokens

    Input: x: (batch_size, T, R)
    Output: logits: (batch_size, num_classes)
    """

    def __init__(
        self,
        n_rois: int = 90,
        d_model: int = 16,
        temporal_hidden_channels: int = 32,
        temporal_kernel_size: int = 3,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 3,
        transformer_dim_feedforward: int = 256,
        transformer_dropout: float = 0.1,
        num_classes: int = 1,
    ):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(
            n_rois=n_rois,
            d_model=d_model,
            hidden_channels=temporal_hidden_channels,
            kernel_size=temporal_kernel_size,
        )

        self.spatial_transformer = ROITransformer(
            n_rois=n_rois,
            d_model=d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, R) where T=176 after dataloader padding/truncation
        roi_embeddings = self.temporal_encoder(x)   # (B, R, d_model)
        logits = self.spatial_transformer(roi_embeddings)
        return logits
