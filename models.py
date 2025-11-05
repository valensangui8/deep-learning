import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class NetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        # input: (B,1,36,36)
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(24 * 6 * 6, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 9 * 9, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # 36x36 -> after two pools: 9x9 (with padding convs)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Registry of available models
MODEL_REGISTRY = {
    "small": NetSmall,
    "baseline": NetBaseline,
    "large": NetLarge,
}


class NetTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NetBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # 36x36 -> pool -> 18x18 -> pool -> 9x9
        self.fc1 = nn.Linear(32 * 9 * 9, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NetThreeConv(nn.Module):
    def __init__(self):
        super().__init__()
        # keep size with padding, pool after first two conv blocks
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 36 -> pool -> 18 -> conv -> pool -> 9
        self.fc1 = nn.Linear(48 * 9 * 9, 128)
        self.dropout = nn.Dropout(p=0.35)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class NetResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 36 -> 18
        self.conv_mid = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.block2 = ResidualBlock(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 18 -> 9
        self.fc1 = nn.Linear(64 * 9 * 9, 128)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.block1(x)
        x = self.pool1(x)
        x = F.relu(self.conv_mid(x))
        x = self.block2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# 5 NEW MODELS BASED ON PROFESSOR'S RECOMMENDATIONS
# ============================================================

# 1. ResNet18 Pretrained + Fine-tuning
class ResNet18Pretrained(nn.Module):
    """ResNet18 pretrained on ImageNet with fine-tuning for face detection"""
    def __init__(self, input_size=224):
        super().__init__()
        self.input_size = input_size
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Data loader already converts grayscale to RGB (3 channels)
        # Replace final layer for binary classification
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.backbone = resnet
        
    def forward(self, x):
        x = self.backbone(x)
        return x


# 2. MobileNetV2 Pretrained + Fine-tuning
class MobileNetV2Pretrained(nn.Module):
    """MobileNetV2 pretrained on ImageNet with fine-tuning for face detection"""
    def __init__(self, input_size=224):
        super().__init__()
        self.input_size = input_size
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Data loader already converts grayscale to RGB (3 channels)
        # Replace classifier
        num_features = mobilenet.classifier[1].in_features
        mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        self.backbone = mobilenet
        
    def forward(self, x):
        x = self.backbone(x)
        return x


# 3. EfficientNet-B0 Pretrained
class EfficientNetB0Pretrained(nn.Module):
    """EfficientNet-B0 pretrained on ImageNet with fine-tuning"""
    def __init__(self, input_size=224):
        super().__init__()
        self.input_size = input_size
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Data loader already converts grayscale to RGB (3 channels)
        # Replace classifier
        num_features = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.backbone = efficientnet
        
    def forward(self, x):
        x = self.backbone(x)
        return x


# 4. Improved CNN Architecture with best practices
class ImprovedCNN(nn.Module):
    """Improved CNN with better architecture: BatchNorm, deeper layers, better regularization"""
    def __init__(self):
        super().__init__()
        # Input: (B, 1, 36, 36)
        # Block 1: 36x36 -> 18x18
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        # Block 2: 18x18 -> 9x9
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)
        
        # Block 3: 9x9 -> 4x4 (optional, for more depth)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        return x


# 5. CNN with Attention Mechanisms
class AttentionBlock(nn.Module):
    """Channel Attention Block (similar to SE-Net)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Avg pooling path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        # Max pooling path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        # Combine
        out = avg_out + max_out
        return x * out


class CNNWithAttention(nn.Module):
    """CNN with attention mechanisms for better feature focus"""
    def __init__(self):
        super().__init__()
        # Block 1: 36x36 -> 18x18
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.attention1 = AttentionBlock(32, reduction=8)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 18x18 -> 9x9
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.attention2 = AttentionBlock(64, reduction=8)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 9x9 -> 4x4
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.attention3 = AttentionBlock(128, reduction=16)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention1(x)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.attention2(x)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.attention3(x)
        x = self.pool3(x)
        
        # FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# Extend registry with new variants
MODEL_REGISTRY.update({
    "tiny": NetTiny,
    "bn": NetBatchNorm,
    "threeconv": NetThreeConv,
    "residual": NetResidual,
    # New models based on professor's recommendations
    "resnet18": ResNet18Pretrained,
    "mobilenetv2": MobileNetV2Pretrained,
    "efficientnet": EfficientNetB0Pretrained,
    "improved": ImprovedCNN,
    "attention": CNNWithAttention,
})


