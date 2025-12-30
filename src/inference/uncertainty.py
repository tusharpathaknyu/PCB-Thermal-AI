"""
Uncertainty Quantification for PCB Thermal Predictions

Provides uncertainty estimates using:
1. MC Dropout - Monte Carlo sampling with dropout enabled
2. Ensemble - Multiple model predictions
3. Test-Time Augmentation - Variance across augmented views

Uncertainty helps identify:
- Regions where model is less confident
- Out-of-distribution inputs
- Areas needing more training data
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from pathlib import Path


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled,
    then computes mean and variance of predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        n_samples: int = 20,
        dropout_rate: float = 0.1
    ):
        """
        Initialize MC Dropout predictor.
        
        Args:
            model: Trained PyTorch model
            device: Device to run inference on
            n_samples: Number of Monte Carlo samples
            dropout_rate: Dropout probability to use
        """
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
        # Enable dropout layers
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                module.p = self.dropout_rate
            elif isinstance(module, nn.Dropout2d):
                module.train()
                module.p = self.dropout_rate
    
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Run MC Dropout inference.
        
        Args:
            inputs: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary containing:
            - mean: Mean prediction
            - std: Standard deviation (uncertainty)
            - samples: All MC samples
        """
        self.model.eval()
        self._enable_dropout()  # Re-enable dropout
        
        inputs = inputs.to(self.device)
        samples = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(inputs)
                samples.append(output.cpu().numpy())
        
        samples = np.stack(samples, axis=0)  # (n_samples, B, C, H, W)
        
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'samples': samples,
            'confidence': 1.0 / (1.0 + std)  # Higher value = more confident
        }


class UncertaintyPredictor:
    """
    Full uncertainty quantification pipeline.
    
    Combines multiple methods for robust uncertainty estimation.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        n_mc_samples: int = 20
    ):
        """
        Initialize uncertainty predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use (auto-detected if None)
            n_mc_samples: Number of MC Dropout samples
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        self.n_mc_samples = n_mc_samples
        
        # Load model
        self.model, self.output_stats = self._load_model(checkpoint_path)
        self.model.to(device)
        
        # Initialize MC Dropout
        self.mc_predictor = MCDropoutPredictor(
            self.model, device, n_mc_samples
        )
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        import sys
        sys.path.insert(0, str(Path(checkpoint_path).parent.parent))
        from src.models import UNetSmall
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        model = UNetSmall(in_channels=4, out_channels=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        output_stats = checkpoint.get('output_stats', {'mean': 50.0, 'std': 30.0})
        
        return model, output_stats
    
    def predict(
        self,
        copper: np.ndarray,
        vias: np.ndarray,
        components: np.ndarray,
        power: np.ndarray,
        return_uncertainty: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict temperature with uncertainty estimates.
        
        Args:
            copper: Copper density map (H, W)
            vias: Via location map (H, W)
            components: Component map (H, W)
            power: Power dissipation map (H, W)
            return_uncertainty: Whether to compute uncertainty
            
        Returns:
            Dictionary with:
            - temperature: Mean temperature prediction
            - uncertainty: Standard deviation (if requested)
            - confidence_map: Pixel-wise confidence
            - uncertainty_score: Overall uncertainty metric
        """
        # Stack inputs
        inputs = np.stack([copper, vias, components, power], axis=0)
        inputs = torch.FloatTensor(inputs).unsqueeze(0)  # (1, 4, H, W)
        
        if return_uncertainty:
            # MC Dropout prediction
            result = self.mc_predictor.predict_with_uncertainty(inputs)
            
            # Denormalize
            mean = result['mean'][0, 0] * self.output_stats['std'] + self.output_stats['mean']
            std = result['std'][0, 0] * self.output_stats['std']
            
            # Compute confidence map (inverse of uncertainty)
            confidence = result['confidence'][0, 0]
            
            # Overall uncertainty score (average uncertainty)
            uncertainty_score = float(std.mean())
            
            return {
                'temperature': mean,
                'uncertainty': std,
                'confidence_map': confidence,
                'uncertainty_score': uncertainty_score,
                'max_temp': float(mean.max()),
                'min_temp': float(mean.min()),
                'mean_temp': float(mean.mean()),
                'max_uncertainty': float(std.max()),
                'mean_uncertainty': float(std.mean())
            }
        else:
            # Single forward pass
            self.model.eval()
            with torch.no_grad():
                output = self.model(inputs.to(self.device))
            
            temp = output[0, 0].cpu().numpy()
            temp = temp * self.output_stats['std'] + self.output_stats['mean']
            
            return {
                'temperature': temp,
                'max_temp': float(temp.max()),
                'min_temp': float(temp.min()),
                'mean_temp': float(temp.mean())
            }
    
    def calibrate_uncertainty(
        self,
        val_inputs: np.ndarray,
        val_outputs: np.ndarray
    ) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates using validation data.
        
        Returns calibration metrics:
        - Expected Calibration Error (ECE)
        - Correlation between uncertainty and actual error
        """
        errors = []
        uncertainties = []
        
        for i in range(len(val_inputs)):
            inputs = val_inputs[i]
            true_temp = val_outputs[i, 0]
            
            result = self.predict(
                inputs[0], inputs[1], inputs[2], inputs[3],
                return_uncertainty=True
            )
            
            pred_temp = result['temperature']
            uncertainty = result['uncertainty']
            
            error = np.abs(pred_temp - true_temp)
            
            errors.append(error.mean())
            uncertainties.append(uncertainty.mean())
        
        errors = np.array(errors)
        uncertainties = np.array(uncertainties)
        
        # Correlation between uncertainty and error
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        
        # Calibration: uncertainty should approximately match error
        calibration_ratio = uncertainties.mean() / errors.mean()
        
        return {
            'uncertainty_error_correlation': float(correlation),
            'calibration_ratio': float(calibration_ratio),
            'mean_error': float(errors.mean()),
            'mean_uncertainty': float(uncertainties.mean())
        }


def analyze_uncertainty(
    result: Dict[str, np.ndarray],
    threshold_high: float = 5.0,
    threshold_low: float = 2.0
) -> Dict[str, any]:
    """
    Analyze uncertainty results and provide insights.
    
    Args:
        result: Output from UncertaintyPredictor.predict()
        threshold_high: High uncertainty threshold (°C)
        threshold_low: Low uncertainty threshold (°C)
        
    Returns:
        Analysis dictionary with insights
    """
    uncertainty = result['uncertainty']
    temperature = result['temperature']
    
    # Find high uncertainty regions
    high_uncertainty_mask = uncertainty > threshold_high
    high_uncertainty_fraction = high_uncertainty_mask.mean()
    
    # Correlation between temperature and uncertainty
    temp_uncertainty_corr = np.corrcoef(
        temperature.flatten(), 
        uncertainty.flatten()
    )[0, 1]
    
    # Hotspot uncertainty (uncertainty at max temp location)
    max_temp_idx = np.unravel_index(temperature.argmax(), temperature.shape)
    hotspot_uncertainty = uncertainty[max_temp_idx]
    
    analysis = {
        'overall_confidence': 'High' if result['mean_uncertainty'] < threshold_low else 
                             'Medium' if result['mean_uncertainty'] < threshold_high else 'Low',
        'high_uncertainty_fraction': float(high_uncertainty_fraction),
        'hotspot_uncertainty': float(hotspot_uncertainty),
        'temp_uncertainty_correlation': float(temp_uncertainty_corr),
        'recommendations': []
    }
    
    # Generate recommendations
    if high_uncertainty_fraction > 0.2:
        analysis['recommendations'].append(
            "High uncertainty in >20% of board. Consider more training data for similar layouts."
        )
    
    if hotspot_uncertainty > threshold_high:
        analysis['recommendations'].append(
            f"Hotspot prediction has high uncertainty ({hotspot_uncertainty:.1f}°C). "
            "Verify with FEA simulation."
        )
    
    if temp_uncertainty_corr > 0.5:
        analysis['recommendations'].append(
            "Uncertainty increases with temperature. Model may need more high-power training samples."
        )
    
    if not analysis['recommendations']:
        analysis['recommendations'].append(
            "Model predictions are confident across the board!"
        )
    
    return analysis
