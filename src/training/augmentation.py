"""
Data Augmentation for PCB Thermal Dataset

Applies geometric and intensity augmentations to increase 
effective dataset size and improve model generalization.

Augmentations:
- Rotation (90°, 180°, 270°)
- Horizontal/Vertical flip
- Random crop and resize
- Noise injection
- Power scaling
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict
import random


class ThermalAugmentation:
    """
    Augmentation pipeline for PCB thermal data.
    
    Applies identical transforms to both input features and output temperature.
    """
    
    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        noise: bool = True,
        power_scale: bool = True,
        crop: bool = False,
        p: float = 0.5
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            rotation: Enable 90° rotation augmentations
            flip: Enable horizontal/vertical flips
            noise: Enable noise injection
            power_scale: Enable power map scaling
            crop: Enable random crop and resize
            p: Probability of applying each augmentation
        """
        self.rotation = rotation
        self.flip = flip
        self.noise = noise
        self.power_scale = power_scale
        self.crop = crop
        self.p = p
    
    def __call__(
        self, 
        inputs: np.ndarray, 
        outputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations to input-output pair.
        
        Args:
            inputs: Input features (C, H, W) - copper, vias, components, power
            outputs: Output temperature (1, H, W)
            
        Returns:
            Augmented (inputs, outputs) tuple
        """
        # Convert to numpy if needed
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.numpy()
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.numpy()
        
        # Make copies
        inputs = inputs.copy()
        outputs = outputs.copy()
        
        # Geometric augmentations (apply same to inputs and outputs)
        if self.rotation and random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90°, 180°, 270°
            inputs = np.rot90(inputs, k, axes=(1, 2)).copy()
            outputs = np.rot90(outputs, k, axes=(1, 2)).copy()
        
        if self.flip:
            if random.random() < self.p:
                inputs = np.flip(inputs, axis=1).copy()  # Vertical flip
                outputs = np.flip(outputs, axis=1).copy()
            if random.random() < self.p:
                inputs = np.flip(inputs, axis=2).copy()  # Horizontal flip
                outputs = np.flip(outputs, axis=2).copy()
        
        # Intensity augmentations
        if self.noise and random.random() < self.p:
            # Add small noise to inputs (except binary masks)
            noise_level = random.uniform(0.01, 0.05)
            inputs[0] += np.random.normal(0, noise_level, inputs[0].shape)  # Copper
            inputs[0] = np.clip(inputs[0], 0, 1)
        
        if self.power_scale and random.random() < self.p:
            # Scale power map (and temperature output accordingly)
            scale = random.uniform(0.8, 1.2)
            inputs[3] *= scale  # Power channel
            
            # Temperature scales linearly with power (approximately)
            # T - T_ambient scales with power, so adjust the delta
            t_ambient = 25.0
            outputs = t_ambient + (outputs - t_ambient) * scale
        
        if self.crop and random.random() < self.p:
            # Random crop and resize
            h, w = inputs.shape[1], inputs.shape[2]
            crop_size = random.randint(int(0.7 * h), int(0.9 * h))
            
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            
            inputs_crop = inputs[:, y:y+crop_size, x:x+crop_size]
            outputs_crop = outputs[:, y:y+crop_size, x:x+crop_size]
            
            # Resize back to original size
            from scipy.ndimage import zoom
            scale_h = h / crop_size
            scale_w = w / crop_size
            
            inputs = np.stack([
                zoom(inputs_crop[c], (scale_h, scale_w), order=1)
                for c in range(inputs_crop.shape[0])
            ])
            outputs = np.stack([
                zoom(outputs_crop[c], (scale_h, scale_w), order=1)
                for c in range(outputs_crop.shape[0])
            ])
        
        return inputs, outputs
    
    def __repr__(self):
        return (f"ThermalAugmentation(rotation={self.rotation}, flip={self.flip}, "
                f"noise={self.noise}, power_scale={self.power_scale}, crop={self.crop}, p={self.p})")


class OnlineAugmentation:
    """
    Online augmentation that generates multiple views of same sample.
    Useful for test-time augmentation (TTA) for better predictions.
    """
    
    def __init__(self, num_augmentations: int = 4):
        self.num_augmentations = num_augmentations
    
    def get_views(self, inputs: np.ndarray) -> list:
        """Generate multiple augmented views of input."""
        views = [inputs]  # Original
        
        # Add rotated versions
        for k in [1, 2, 3]:
            rotated = np.rot90(inputs, k, axes=(1, 2)).copy()
            views.append(rotated)
        
        # Add flipped versions
        views.append(np.flip(inputs, axis=1).copy())
        views.append(np.flip(inputs, axis=2).copy())
        
        return views[:self.num_augmentations]
    
    def inverse_transform(self, outputs: list) -> list:
        """Inverse transform augmented outputs back to original orientation."""
        if len(outputs) == 1:
            return outputs
        
        result = [outputs[0]]  # Original stays same
        
        # Inverse rotations
        if len(outputs) > 1:
            result.append(np.rot90(outputs[1], -1, axes=(1, 2)))  # -90°
        if len(outputs) > 2:
            result.append(np.rot90(outputs[2], -2, axes=(1, 2)))  # -180°
        if len(outputs) > 3:
            result.append(np.rot90(outputs[3], -3, axes=(1, 2)))  # -270°
        
        # Inverse flips
        if len(outputs) > 4:
            result.append(np.flip(outputs[4], axis=1))
        if len(outputs) > 5:
            result.append(np.flip(outputs[5], axis=2))
        
        return result
    
    def aggregate(self, outputs: list, method: str = 'mean') -> np.ndarray:
        """Aggregate multiple predictions."""
        # First inverse transform all outputs
        aligned = self.inverse_transform(outputs)
        
        stacked = np.stack(aligned)
        
        if method == 'mean':
            return stacked.mean(axis=0)
        elif method == 'median':
            return np.median(stacked, axis=0)
        elif method == 'max':
            return stacked.max(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


# Augmentation presets
def get_augmentation(preset: str = 'default') -> ThermalAugmentation:
    """Get predefined augmentation configuration."""
    presets = {
        'none': ThermalAugmentation(
            rotation=False, flip=False, noise=False, 
            power_scale=False, crop=False, p=0
        ),
        'light': ThermalAugmentation(
            rotation=True, flip=True, noise=False,
            power_scale=False, crop=False, p=0.3
        ),
        'default': ThermalAugmentation(
            rotation=True, flip=True, noise=True,
            power_scale=True, crop=False, p=0.5
        ),
        'heavy': ThermalAugmentation(
            rotation=True, flip=True, noise=True,
            power_scale=True, crop=True, p=0.7
        )
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]
