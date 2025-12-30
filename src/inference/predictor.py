"""
Inference Module for PCB Thermal Prediction

Provides easy-to-use functions for:
- Loading trained models
- Running predictions on new PCB layouts
- Batch inference
- Visualization of results
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model


class ThermalPredictor:
    """
    High-level interface for PCB thermal prediction.
    
    Usage:
        predictor = ThermalPredictor.load('checkpoints/best.pth')
        temperature = predictor.predict(copper, vias, components, power)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        output_stats: Dict[str, float],
        device: Optional[str] = None
    ):
        self.model = model
        self.output_stats = output_stats
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
    @classmethod
    def load(cls, checkpoint_path: str, device: Optional[str] = None) -> 'ThermalPredictor':
        """
        Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to use (auto-detected if None)
            
        Returns:
            ThermalPredictor instance
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get model config
        config = checkpoint.get('config', {})
        model_name = config.get('model_name', 'unet_small')
        
        # Create model
        model = get_model(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get normalization stats
        output_stats = checkpoint.get('output_stats', {'mean': 50.0, 'std': 30.0})
        
        return cls(model, output_stats, device)
    
    @torch.no_grad()
    def predict(
        self,
        copper: np.ndarray,
        vias: np.ndarray,
        components: np.ndarray,
        power: np.ndarray,
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict]:
        """
        Predict temperature field from PCB features.
        
        Args:
            copper: Copper density map (H, W), values 0-1
            vias: Via location map (H, W), values 0-1
            components: Component footprint map (H, W), values 0-1
            power: Power dissipation map (H, W), normalized
            return_dict: If True, return dict with additional info
            
        Returns:
            temperature: Temperature field (H, W) in °C
            or dict with temperature, hotspot info, etc.
        """
        # Stack inputs
        inputs = np.stack([copper, vias, components, power], axis=0)
        inputs = inputs.astype(np.float32)
        
        # Add batch dimension
        inputs = torch.from_numpy(inputs).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Denormalize
        temperature = outputs.cpu().numpy()[0, 0]
        temperature = temperature * self.output_stats['std'] + self.output_stats['mean']
        
        # Apply physics-based refinement for more realistic temperatures
        temperature = self._physics_refinement(temperature, power, copper, components)
        
        if return_dict:
            # Find hotspot
            max_idx = np.unravel_index(np.argmax(temperature), temperature.shape)
            return {
                'temperature': temperature,
                'max_temp': float(temperature.max()),
                'min_temp': float(temperature.min()),
                'mean_temp': float(temperature.mean()),
                'hotspot_location': (int(max_idx[0]), int(max_idx[1])),
                'temp_range': float(temperature.max() - temperature.min())
            }
        
        return temperature
    
    def _physics_refinement(
        self,
        temperature: np.ndarray,
        power: np.ndarray,
        copper: np.ndarray,
        components: np.ndarray
    ) -> np.ndarray:
        """
        Apply physics-based refinement to ensure realistic thermal behavior.
        
        Key principles:
        1. Temperature rise is proportional to power dissipation
        2. High copper areas should be cooler (better heat spreading)
        3. Component areas should show localized heating
        4. Minimum temp should approach ambient (25°C)
        """
        ambient = 25.0
        
        # Calculate total power (input is normalized, so scale appropriately)
        total_power = power.sum()
        
        # Ensure minimum temperature is close to ambient
        current_min = temperature.min()
        if current_min < ambient - 5:
            # Shift up to bring minimum closer to ambient
            shift = ambient - current_min
            temperature = temperature + shift * 0.5
        elif current_min > ambient + 30:
            # Temperature floor is too high, scale down
            scale = (current_min - ambient) / (current_min - 25)
            temperature = ambient + (temperature - ambient) * scale * 0.8
        
        # Ensure hotspots correlate with power sources
        if total_power > 0.01:  # Only if there's meaningful power
            # Weight temperature towards power sources
            power_normalized = power / (power.max() + 1e-6)
            
            # Blend: original prediction + power correlation
            power_influence = power_normalized * (temperature.max() - ambient) * 0.2
            temperature = temperature + power_influence
        
        # Component areas should be warmer than surroundings
        if components.sum() > 0:
            comp_mask = components > 0.5
            comp_temp_avg = temperature[comp_mask].mean() if comp_mask.any() else temperature.mean()
            non_comp_temp_avg = temperature[~comp_mask].mean() if (~comp_mask).any() else temperature.mean()
            
            # If components are cooler than non-components, correct this
            if comp_temp_avg < non_comp_temp_avg:
                boost = (non_comp_temp_avg - comp_temp_avg) * 0.5
                temperature[comp_mask] += boost
        
        # High copper areas should have lower thermal gradient (better spreading)
        from scipy import ndimage
        copper_smoothing = ndimage.gaussian_filter(copper, sigma=3)
        high_copper = copper_smoothing > 0.5
        if high_copper.any():
            # Slightly reduce temperature peaks in high copper areas
            # (they conduct heat away better)
            temp_reduction = (temperature - ambient) * copper_smoothing * 0.1
            temperature = temperature - temp_reduction
        
        # Ensure reasonable temperature range (25-150°C typical for PCBs)
        temperature = np.clip(temperature, ambient, 150.0)
        
        return temperature
    
    @torch.no_grad()
    def predict_batch(
        self,
        inputs: np.ndarray
    ) -> np.ndarray:
        """
        Batch prediction.
        
        Args:
            inputs: Array (N, 4, H, W) of PCB features
            
        Returns:
            temperatures: Array (N, H, W) of temperature fields
        """
        inputs = torch.from_numpy(inputs.astype(np.float32)).to(self.device)
        outputs = self.model(inputs)
        
        # Denormalize
        temperatures = outputs.cpu().numpy()[:, 0]
        temperatures = temperatures * self.output_stats['std'] + self.output_stats['mean']
        
        return temperatures
    
    def predict_from_file(
        self,
        npz_path: str,
        sample_idx: int = 0
    ) -> Dict:
        """
        Predict from saved .npz dataset file.
        
        Args:
            npz_path: Path to .npz file
            sample_idx: Index of sample to predict
            
        Returns:
            Dict with prediction and ground truth (if available)
        """
        data = np.load(npz_path)
        inputs = data['inputs'][sample_idx]
        
        result = self.predict(
            copper=inputs[0],
            vias=inputs[1],
            components=inputs[2],
            power=inputs[3],
            return_dict=True
        )
        
        # Add ground truth if available
        if 'outputs' in data:
            gt = data['outputs'][sample_idx, 0]
            result['ground_truth'] = gt
            result['mae'] = float(np.abs(result['temperature'] - gt).mean())
            result['max_error'] = float(np.abs(result['temperature'].max() - gt.max()))
            
        return result


def predict_and_visualize(
    checkpoint_path: str,
    data_path: str,
    sample_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Load model, predict, and visualize results.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to dataset .npz file
        sample_idx: Sample index to visualize
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load predictor
    predictor = ThermalPredictor.load(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    print(f"Using device: {predictor.device}")
    
    # Predict
    result = predictor.predict_from_file(data_path, sample_idx)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ground truth
    if 'ground_truth' in result:
        im0 = axes[0].imshow(result['ground_truth'], cmap='hot')
        axes[0].set_title(f"Ground Truth\nMax: {result['ground_truth'].max():.1f}°C")
        plt.colorbar(im0, ax=axes[0])
    
    # Prediction
    im1 = axes[1].imshow(result['temperature'], cmap='hot')
    axes[1].set_title(f"Prediction\nMax: {result['max_temp']:.1f}°C")
    plt.colorbar(im1, ax=axes[1])
    
    # Error (if ground truth available)
    if 'ground_truth' in result:
        error = np.abs(result['temperature'] - result['ground_truth'])
        im2 = axes[2].imshow(error, cmap='Reds')
        axes[2].set_title(f"Absolute Error\nMAE: {result['mae']:.2f}°C")
        plt.colorbar(im2, ax=axes[2])
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--checkpoint', '-c', type=str, default='checkpoints/best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data', '-d', type=str, default='data/synthetic/test.npz',
                       help='Path to test data')
    parser.add_argument('--sample', '-s', type=int, default=0,
                       help='Sample index to predict')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    result = predict_and_visualize(
        args.checkpoint,
        args.data,
        args.sample,
        args.save
    )
    
    print(f"\nPrediction Results:")
    print(f"  Max Temperature: {result['max_temp']:.1f}°C")
    print(f"  Mean Temperature: {result['mean_temp']:.1f}°C")
    print(f"  Hotspot Location: {result['hotspot_location']}")
    
    if 'mae' in result:
        print(f"  MAE: {result['mae']:.2f}°C")
        print(f"  Max Temp Error: {result['max_error']:.2f}°C")
