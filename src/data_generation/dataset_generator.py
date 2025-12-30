"""
Dataset Generator for PCB Thermal AI

Combines PCB layout generation with thermal solving to create
training datasets. Outputs can be saved in multiple formats:
- NumPy (.npz)
- HDF5 (.h5) for larger datasets
- Image pairs (PNG) for visualization

Dataset structure:
- Input: (N, C, H, W) where C = [copper, vias, components, power]
- Output: (N, 1, H, W) temperature field
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import json
import os

from .pcb_generator import PCBGenerator, PCBLayout
from .thermal_solver import HeatEquationSolver, ThermalProperties


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    # Grid settings
    grid_size: Tuple[int, int] = (128, 128)
    physical_size: Tuple[float, float] = (0.1, 0.1)  # 10cm x 10cm
    
    # Dataset size
    num_samples: int = 1000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Complexity distribution
    simple_ratio: float = 0.3
    medium_ratio: float = 0.5
    complex_ratio: float = 0.2
    
    # Thermal properties variation
    vary_ambient: bool = True
    ambient_range: Tuple[float, float] = (20.0, 35.0)
    vary_convection: bool = True
    convection_range: Tuple[float, float] = (5.0, 25.0)
    
    # Output format
    output_format: str = "npz"  # "npz", "h5", or "both"
    save_metadata: bool = True
    

class DatasetGenerator:
    """
    Generates complete training datasets for PCB thermal prediction.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.pcb_gen = PCBGenerator(grid_size=self.config.grid_size)
        
    def generate_sample(
        self,
        complexity: str,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict]:
        """
        Generate a single sample (input features + temperature output).
        
        Returns:
            inputs: Dict with 'copper', 'vias', 'components', 'power' arrays
            temperature: 2D temperature field
            metadata: Dict with sample information
        """
        if seed is not None:
            self.pcb_gen.rng = np.random.default_rng(seed)
            
        # Generate PCB layout
        layout = self.pcb_gen.generate(complexity=complexity)
        
        # Set up thermal solver with possibly varied properties
        props = ThermalProperties()
        
        if self.config.vary_ambient:
            props.t_ambient = np.random.uniform(*self.config.ambient_range)
        if self.config.vary_convection:
            props.h_conv = np.random.uniform(*self.config.convection_range)
            
        solver = HeatEquationSolver(
            grid_size=self.config.grid_size,
            physical_size=self.config.physical_size,
            properties=props
        )
        
        # Set copper density (affects thermal conductivity)
        solver.set_copper_density(layout.copper_density)
        
        # Add heat sources from components
        for comp in layout.components:
            solver.add_heat_source(
                center=(comp.center_y, comp.center_x),
                size=(comp.height, comp.width),
                power=comp.power
            )
            
        # Solve for temperature
        temperature = solver.solve()
        layout.temperature = temperature
        
        # Prepare inputs
        inputs = {
            'copper': layout.copper_density,
            'vias': layout.via_map,
            'components': layout.component_map,
            'power': layout.power_map
        }
        
        # Metadata
        hotspot_info = solver.get_hotspot_info(temperature)
        metadata = {
            'complexity': complexity,
            'num_components': len(layout.components),
            'total_power': sum(c.power for c in layout.components),
            'num_vias': len(layout.vias),
            'num_thermal_vias': sum(1 for v in layout.vias if v.is_thermal),
            'copper_fill': float(layout.copper_density.mean()),
            't_ambient': props.t_ambient,
            'h_conv': props.h_conv,
            **hotspot_info
        }
        
        return inputs, temperature, metadata
    
    def generate_dataset(
        self,
        output_dir: str,
        num_samples: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Generate complete dataset and save to disk.
        
        Args:
            output_dir: Directory to save dataset
            num_samples: Override number of samples
            show_progress: Show progress bar
            
        Returns:
            Statistics about generated dataset
        """
        num_samples = num_samples or self.config.num_samples
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate split sizes
        n_train = int(num_samples * self.config.train_ratio)
        n_val = int(num_samples * self.config.val_ratio)
        n_test = num_samples - n_train - n_val
        
        # Calculate complexity distribution
        complexities = (
            ['simple'] * int(num_samples * self.config.simple_ratio) +
            ['medium'] * int(num_samples * self.config.medium_ratio) +
            ['complex'] * int(num_samples * self.config.complex_ratio)
        )
        # Pad or trim to exact num_samples
        while len(complexities) < num_samples:
            complexities.append('medium')
        complexities = complexities[:num_samples]
        np.random.shuffle(complexities)
        
        # Storage arrays
        h, w = self.config.grid_size
        all_inputs = np.zeros((num_samples, 4, h, w), dtype=np.float32)
        all_outputs = np.zeros((num_samples, 1, h, w), dtype=np.float32)
        all_metadata = []
        
        # Generate samples
        iterator = range(num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating samples")
            
        for i in iterator:
            inputs, temperature, metadata = self.generate_sample(
                complexity=complexities[i],
                seed=i  # Reproducible
            )
            
            # Stack inputs: (4, H, W)
            all_inputs[i, 0] = inputs['copper']
            all_inputs[i, 1] = inputs['vias']
            all_inputs[i, 2] = inputs['components']
            all_inputs[i, 3] = inputs['power'] / 1000  # Normalize power
            
            # Output: (1, H, W)
            all_outputs[i, 0] = temperature
            
            metadata['sample_id'] = i
            all_metadata.append(metadata)
            
        # Split into train/val/test
        indices = np.random.permutation(num_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        splits = {
            'train': (all_inputs[train_idx], all_outputs[train_idx], 
                     [all_metadata[i] for i in train_idx]),
            'val': (all_inputs[val_idx], all_outputs[val_idx],
                   [all_metadata[i] for i in val_idx]),
            'test': (all_inputs[test_idx], all_outputs[test_idx],
                    [all_metadata[i] for i in test_idx])
        }
        
        # Save based on format
        if self.config.output_format in ['npz', 'both']:
            self._save_npz(splits, output_path)
            
        if self.config.output_format in ['h5', 'both']:
            self._save_h5(splits, output_path)
            
        # Save metadata
        if self.config.save_metadata:
            self._save_metadata(all_metadata, output_path)
            
        # Compute statistics
        stats = self._compute_statistics(all_inputs, all_outputs, all_metadata)
        stats['num_samples'] = num_samples
        stats['splits'] = {'train': n_train, 'val': n_val, 'test': n_test}
        
        # Save stats
        with open(output_path / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        return stats
    
    def _save_npz(self, splits: Dict, output_path: Path):
        """Save dataset in NumPy format"""
        for split_name, (inputs, outputs, _) in splits.items():
            np.savez_compressed(
                output_path / f'{split_name}.npz',
                inputs=inputs,
                outputs=outputs
            )
        print(f"Saved NPZ files to {output_path}")
        
    def _save_h5(self, splits: Dict, output_path: Path):
        """Save dataset in HDF5 format"""
        with h5py.File(output_path / 'dataset.h5', 'w') as f:
            for split_name, (inputs, outputs, _) in splits.items():
                grp = f.create_group(split_name)
                grp.create_dataset('inputs', data=inputs, compression='gzip')
                grp.create_dataset('outputs', data=outputs, compression='gzip')
        print(f"Saved HDF5 file to {output_path}")
        
    def _save_metadata(self, metadata: List[Dict], output_path: Path):
        """Save metadata to JSON"""
        # Convert numpy types to Python types
        clean_metadata = []
        for m in metadata:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean[k] = float(v)
                else:
                    clean[k] = v
            clean_metadata.append(clean)
            
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(clean_metadata, f, indent=2)
            
    def _compute_statistics(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        metadata: List[Dict]
    ) -> Dict:
        """Compute dataset statistics"""
        temps = outputs.flatten()
        powers = [m['total_power'] for m in metadata]
        max_temps = [m['max_temp'] for m in metadata]
        
        return {
            'temperature': {
                'min': float(temps.min()),
                'max': float(temps.max()),
                'mean': float(temps.mean()),
                'std': float(temps.std())
            },
            'power': {
                'min': float(min(powers)),
                'max': float(max(powers)),
                'mean': float(np.mean(powers))
            },
            'max_temps': {
                'min': float(min(max_temps)),
                'max': float(max(max_temps)),
                'mean': float(np.mean(max_temps))
            },
            'complexity_distribution': {
                'simple': sum(1 for m in metadata if m['complexity'] == 'simple'),
                'medium': sum(1 for m in metadata if m['complexity'] == 'medium'),
                'complex': sum(1 for m in metadata if m['complexity'] == 'complex')
            },
            'input_channels': ['copper_density', 'via_map', 'component_map', 'power_map']
        }


def quick_generate(num_samples: int = 100, output_dir: str = "data/synthetic"):
    """Quick function to generate a small dataset"""
    config = DatasetConfig(
        num_samples=num_samples,
        output_format='npz'
    )
    generator = DatasetGenerator(config)
    stats = generator.generate_dataset(output_dir)
    
    print("\n" + "="*50)
    print("Dataset Generation Complete!")
    print("="*50)
    print(f"Samples: {stats['num_samples']}")
    print(f"Splits: {stats['splits']}")
    print(f"Temperature range: {stats['temperature']['min']:.1f}°C - {stats['temperature']['max']:.1f}°C")
    print(f"Power range: {stats['power']['min']:.2f}W - {stats['power']['max']:.2f}W")
    print(f"Output directory: {output_dir}")
    
    return stats


if __name__ == "__main__":
    quick_generate(num_samples=100)
