import torch
import torch.nn as nn


class PhysNetED(nn.Module):
    """
    Encoder–Decoder 3D convolutional network for rPPG waveform prediction.
    Based on the PhysNet architecture from the original BMVC2019 paper repo.:contentReference[oaicite:1]{index=1}
    """

    def __init__(self, in_channels=3):
        super(PhysNetED, self).__init__()

        # ---------------------------------------------------------------------
        # Encoder Path (spatio-temporal feature extraction)
        # ---------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------------
        # Decoder Path (upsampling back to temporal resolution)
        # ---------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------------------
        # Output Layer: Predict a single rPPG channel
        # ---------------------------------------------------------------------
        self.out_conv = nn.Conv3d(32, 1, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, C=3, T, H, W) — video input
        Returns:
            rppg_time: Tensor of shape (B, T) — raw output waveform
        """
        # Encode spatio-temporal features
        features = self.encoder(x)

        # Decode back
        decoded = self.decoder(features)

        # Output layer
        out = self.out_conv(decoded)

        # Spatial global pooling: average across H & W
        # out becomes shape (B, 1, T, 1, 1)
        out = out.mean(dim=-1).mean(dim=-1)  # global average pooling over spatial dims

        # Remove channel dimension -> shape (B, T)
        rppg_time = out.squeeze(1)

        return rppg_time
