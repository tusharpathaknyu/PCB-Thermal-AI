"""
U-Net Model for PCB Thermal Prediction

Architecture designed for predicting temperature fields from PCB layout features.
Based on the original U-Net paper with modifications for thermal prediction:
- Input: 4 channels (copper, vias, components, power)
- Output: 1 channel (temperature field)
- Skip connections to preserve spatial detail
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) x 2"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        mid_channels: Optional[int] = None
    ):
        super().__init__()
        mid_channels = mid_channels or out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: Upsample -> Concat skip -> DoubleConv"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bilinear: bool = True
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, 
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer (to be upsampled)
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (input might not be perfectly divisible)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for PCB thermal prediction.
    
    Architecture:
        Encoder: 4 downsampling blocks (64->128->256->512->1024)
        Decoder: 4 upsampling blocks with skip connections
        
    Args:
        in_channels: Number of input channels (default 4 for PCB features)
        out_channels: Number of output channels (default 1 for temperature)
        features: List of feature sizes for each encoder level
        bilinear: Use bilinear upsampling (vs transposed conv)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        bilinear: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder (downsampling path)
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # Output
        self.outc = OutConv(features[0], out_channels)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 512 (or 1024 if not bilinear)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetSmall(nn.Module):
    """
    Smaller U-Net variant for faster training/inference.
    Good for prototyping and smaller datasets.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
        bilinear: bool = True
    ):
        super().__init__()
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            bilinear=bilinear
        )
        
    def forward(self, x):
        return self.unet(x)
    
    def count_parameters(self) -> int:
        return self.unet.count_parameters()


class ThermalUNet(nn.Module):
    """
    U-Net with thermal-specific modifications:
    - Temperature offset head (predicts T_ambient + delta_T)
    - Optional physics-informed constraints
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_features: int = 64,
        bilinear: bool = True,
        ambient_temp: float = 25.0
    ):
        super().__init__()
        self.ambient_temp = ambient_temp
        
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=1,
            features=[base_features, base_features*2, base_features*4, base_features*8],
            bilinear=bilinear
        )
        
    def forward(self, x):
        # Predict temperature delta (always positive - use ReLU or softplus)
        delta_t = F.softplus(self.unet(x))  # Ensures delta_T >= 0
        
        # Add ambient temperature
        temperature = self.ambient_temp + delta_t
        
        return temperature
    
    def count_parameters(self) -> int:
        return self.unet.count_parameters()


def get_model(model_name: str = "unet", **kwargs) -> nn.Module:
    """
    Factory function to get model by name.
    
    Args:
        model_name: One of "unet", "unet_small", "thermal_unet"
        **kwargs: Additional arguments for model
        
    Returns:
        Model instance
    """
    models = {
        "unet": UNet,
        "unet_small": UNetSmall,
        "thermal_unet": ThermalUNet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
        
    return models[model_name](**kwargs)


def test_model():
    """Test model with dummy input"""
    print("Testing U-Net models...")
    
    # Test input: batch of 2, 4 channels, 128x128
    x = torch.randn(2, 4, 128, 128)
    
    for name in ["unet", "unet_small", "thermal_unet"]:
        model = get_model(name)
        y = model(x)
        
        print(f"\n{name}:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Parameters:   {model.count_parameters():,}")
        
    print("\nAll models passed!")


if __name__ == "__main__":
    test_model()
