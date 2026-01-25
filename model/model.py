"""
Phone-optimized neural network for deepfake detection.
Designed for 8kHz, 64-mel input from phone calls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhoneCNN(nn.Module):
    """
    Lightweight CNN optimized for phone-quality audio.

    Key differences from standard audio CNNs:
    - Smaller input (64 mels vs 128)
    - Optimized for narrow frequency band
    - Focus on temporal patterns (AI artifacts)
    - Small enough for mobile deployment
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 64x94 -> 32x47
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x47 -> 16x23
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16x23 -> 8x11
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 8x11 -> 4x4 (adaptive)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class TemporalFocusCNN(nn.Module):
    """
    CNN that focuses on temporal patterns.

    AI-generated audio often has subtle timing artifacts:
    - Unnatural pauses
    - Too-perfect timing
    - Repeated micro-patterns

    This architecture uses 1D convolutions along time axis
    to catch these patterns.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Frequency feature extraction (2D conv)
        self.freq_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )

        # Temporal feature extraction (1D conv along time)
        self.time_features = nn.Sequential(
            nn.Conv1d(64 * 16, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Extract frequency features
        x = self.freq_features(x)  # (B, 64, 16, T)

        # Reshape for temporal processing
        x = x.view(batch_size, -1, x.shape[-1])  # (B, 64*16, T)

        # Extract temporal features
        x = self.time_features(x)  # (B, 256, 8)

        # Classify
        x = self.classifier(x)
        return x


class MobilePhoneNet(nn.Module):
    """
    Ultra-lightweight model for mobile deployment.
    Uses depthwise separable convolutions for efficiency.

    Target: <5MB model size, <100ms inference on phone.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Depthwise separable blocks
            self._dsconv(16, 32, stride=2),
            self._dsconv(32, 64, stride=2),
            self._dsconv(64, 128, stride=2),
            self._dsconv(128, 128, stride=1),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def _dsconv(self, in_ch: int, out_ch: int, stride: int = 1):
        """Depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class PhoneDeepfakeDetector(nn.Module):
    """
    Full-featured detector with attention mechanism.
    Best accuracy but larger model.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Self-attention
        self.attention = SelfAttention(128)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x


class SelfAttention(nn.Module):
    """Self-attention block for capturing global patterns."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x


def get_model(model_name: str = "phone_cnn", num_classes: int = 2) -> nn.Module:
    """Factory function to get model by name."""

    models = {
        "phone_cnn": PhoneCNN,
        "temporal": TemporalFocusCNN,
        "mobile": MobilePhoneNet,
        "attention": PhoneDeepfakeDetector,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model: nn.Module) -> float:
    """Estimate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


if __name__ == "__main__":
    # Test all models
    print("Testing phone-optimized models...")
    print("-" * 60)

    # Phone audio input: (batch, 1, 64 mels, ~94 time frames)
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 94)

    for name in ["phone_cnn", "temporal", "mobile", "attention"]:
        model = get_model(name)
        output = model(x)
        params = count_parameters(model)
        size_mb = estimate_model_size(model)

        print(f"{name:12} | Output: {output.shape} | Params: {params:>8,} | Size: {size_mb:.2f} MB")

    print("-" * 60)
    print("All models working!")
