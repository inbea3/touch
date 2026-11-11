"""CNN and prototypical components adapted for tactile classification."""
import torch
from torch import nn
from torch.nn import functional as F

class PressureCNN(nn.Module):
    """Input (B,K,32,32), output class logits."""
    def __init__(self, n_classes: int, k_frames: int = 5, channels: int = 24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(k_frames, channels, 3, padding=1), nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels*2, 3, padding=1), nn.InstanceNorm2d(channels*2, affine=True),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(channels*2, channels*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(channels*4, n_classes)
    def encode(self, x): return self.features(x).flatten(1)
    def forward(self, x): return self.head(self.encode(x))

class SensorVectorEncoder(nn.Module):
    """Encode (B,K,S) by sensor-wise early-window mean and standard deviation."""
    def __init__(self, n_sensors: int, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_sensors*2, 512), nn.ReLU(), nn.Dropout(.15),
                                 nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, embedding_dim))
    def forward(self, x):
        feat = torch.cat((x.mean(1), x.std(1, unbiased=False)), dim=-1)
        return F.normalize(self.net(feat), p=2, dim=-1)

class PrototypicalClassifier(nn.Module):
    """Few-shot episode classifier; support_y must use local labels 0..N-1."""
    def __init__(self, encoder: nn.Module):
        super().__init__(); self.encoder = encoder
    def forward(self, support_x, support_y, query_x, n_way: int):
        support, query = self.encoder(support_x), self.encoder(query_x)
        protos = []
        for cls in range(n_way):
            chosen = support[support_y == cls]
            if chosen.numel() == 0: raise ValueError(f"support episode 缺少类别 {cls}")
            protos.append(chosen.mean(0))
        return -torch.cdist(query, F.normalize(torch.stack(protos), p=2, dim=-1), p=2)
