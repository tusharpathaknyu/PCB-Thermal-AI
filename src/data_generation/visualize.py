"""
Visualization Utilities for PCB Thermal AI

Provides functions to visualize:
- PCB layouts (copper, components, vias)
- Temperature fields
- Input-output pairs for training data
- Dataset statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json


# Custom thermal colormap (blue -> green -> yellow -> red)
THERMAL_COLORS = [
    (0.0, 'blue'),
    (0.25, 'cyan'),
    (0.5, 'green'),
    (0.75, 'yellow'),
    (1.0, 'red')
]
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    'thermal',
    [(pos, color) for pos, color in THERMAL_COLORS]
)


def plot_pcb_layout(
    copper: np.ndarray,
    vias: np.ndarray,
    components: np.ndarray,
    power: np.ndarray,
    title: str = "PCB Layout",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Plot all input channels of a PCB layout.
    
    Args:
        copper: Copper density map (H, W)
        vias: Via location map (H, W)
        components: Component footprint map (H, W)
        power: Power dissipation map (H, W)
        title: Figure title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Copper density
    im0 = axes[0, 0].imshow(copper, cmap='copper', vmin=0, vmax=1)
    axes[0, 0].set_title('Copper Density')
    plt.colorbar(im0, ax=axes[0, 0], label='Fill Ratio')
    
    # Via map
    im1 = axes[0, 1].imshow(vias, cmap='Blues', vmin=0, vmax=1)
    axes[0, 1].set_title('Via Locations (blue=signal, dark=thermal)')
    plt.colorbar(im1, ax=axes[0, 1], label='Via Type')
    
    # Component map
    im2 = axes[1, 0].imshow(components, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('Component Footprints')
    plt.colorbar(im2, ax=axes[1, 0], label='Presence')
    
    # Power map
    im3 = axes[1, 1].imshow(power, cmap='hot')
    axes[1, 1].set_title('Power Dissipation')
    plt.colorbar(im3, ax=axes[1, 1], label='W/m²')
    
    for ax in axes.flat:
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
        
    return fig


def plot_temperature_field(
    temperature: np.ndarray,
    title: str = "Temperature Distribution",
    figsize: Tuple[int, int] = (8, 6),
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Plot temperature field with thermal colormap.
    
    Args:
        temperature: Temperature array (H, W) in °C
        title: Figure title
        figsize: Figure size
        show_colorbar: Whether to show colorbar
        vmin, vmax: Color range limits
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(
        temperature, 
        cmap=THERMAL_CMAP,
        vmin=vmin or temperature.min(),
        vmax=vmax or temperature.max()
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature (°C)')
        
    # Mark hotspot
    max_idx = np.unravel_index(np.argmax(temperature), temperature.shape)
    ax.plot(max_idx[1], max_idx[0], 'k*', markersize=15, 
            label=f'Hotspot: {temperature.max():.1f}°C')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_sample(
    inputs: np.ndarray,
    output: np.ndarray,
    metadata: Optional[Dict] = None,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None
):
    """
    Plot a complete training sample (inputs + output).
    
    Args:
        inputs: Input array (4, H, W) or (H, W, 4)
        output: Output temperature (1, H, W) or (H, W)
        metadata: Optional sample metadata
        figsize: Figure size
        save_path: Optional path to save
    """
    # Handle different input shapes
    if inputs.ndim == 3:
        if inputs.shape[0] == 4:  # (C, H, W)
            copper, vias, components, power = inputs
        else:  # (H, W, C)
            copper = inputs[:, :, 0]
            vias = inputs[:, :, 1]
            components = inputs[:, :, 2]
            power = inputs[:, :, 3]
    
    if output.ndim == 3:
        temperature = output[0]
    else:
        temperature = output
        
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Title with metadata
    title = "Training Sample"
    if metadata:
        title += f" | {metadata.get('complexity', 'N/A')} | "
        title += f"Power: {metadata.get('total_power', 0):.2f}W | "
        title += f"Max: {metadata.get('max_temp', 0):.1f}°C"
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # Inputs
    im0 = axes[0, 0].imshow(copper, cmap='copper', vmin=0, vmax=1)
    axes[0, 0].set_title('Copper Density')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(vias, cmap='Blues')
    axes[0, 1].set_title('Vias')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(components, cmap='Reds')
    axes[0, 2].set_title('Components')
    plt.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[1, 0].imshow(power * 1000, cmap='hot')  # Denormalize
    axes[1, 0].set_title('Power (W/m²)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Output
    im4 = axes[1, 1].imshow(temperature, cmap=THERMAL_CMAP)
    axes[1, 1].set_title('Temperature (°C)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Mark hotspot on temperature
    max_idx = np.unravel_index(np.argmax(temperature), temperature.shape)
    axes[1, 1].plot(max_idx[1], max_idx[0], 'k*', markersize=10)
    
    # Combined overlay
    axes[1, 2].imshow(temperature, cmap=THERMAL_CMAP, alpha=0.7)
    axes[1, 2].contour(copper, levels=[0.3, 0.6, 0.9], colors='white', alpha=0.5)
    axes[1, 2].imshow(components, cmap='Reds', alpha=0.3)
    axes[1, 2].set_title('Overlay (Temp + Copper + Components)')
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_dataset_statistics(
    stats_path: str,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot dataset statistics from saved JSON.
    
    Args:
        stats_path: Path to dataset_stats.json
        figsize: Figure size
        save_path: Optional path to save
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Dataset Statistics', fontsize=14, fontweight='bold')
    
    # Complexity distribution
    complexity = stats['complexity_distribution']
    axes[0, 0].bar(complexity.keys(), complexity.values(), color=['green', 'orange', 'red'])
    axes[0, 0].set_title('Complexity Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # Temperature range
    temp = stats['temperature']
    axes[0, 1].bar(['Min', 'Mean', 'Max'], 
                   [temp['min'], temp['mean'], temp['max']],
                   color=['blue', 'green', 'red'])
    axes[0, 1].set_title('Temperature Statistics (°C)')
    axes[0, 1].set_ylabel('Temperature (°C)')
    
    # Power range
    power = stats['power']
    axes[1, 0].bar(['Min', 'Mean', 'Max'],
                   [power['min'], power['mean'], power['max']],
                   color=['lightblue', 'steelblue', 'darkblue'])
    axes[1, 0].set_title('Power Statistics (W)')
    axes[1, 0].set_ylabel('Power (W)')
    
    # Train/val/test split
    splits = stats['splits']
    axes[1, 1].pie(splits.values(), labels=splits.keys(), autopct='%1.1f%%',
                   colors=['steelblue', 'orange', 'green'])
    axes[1, 1].set_title('Dataset Splits')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def visualize_batch(
    inputs: np.ndarray,
    outputs: np.ndarray,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (16, 4),
    save_path: Optional[str] = None
):
    """
    Visualize multiple samples from a batch.
    
    Args:
        inputs: Batch of inputs (N, 4, H, W)
        outputs: Batch of outputs (N, 1, H, W)
        num_samples: Number of samples to show
        figsize: Figure size
        save_path: Optional path to save
    """
    num_samples = min(num_samples, len(inputs))
    
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    fig.suptitle('Sample Batch: Copper (top) → Temperature (bottom)', fontsize=12)
    
    for i in range(num_samples):
        # Copper (first input channel)
        axes[0, i].imshow(inputs[i, 0], cmap='copper', vmin=0, vmax=1)
        axes[0, i].set_title(f'Sample {i+1}')
        axes[0, i].axis('off')
        
        # Temperature
        temp = outputs[i, 0] if outputs.ndim == 4 else outputs[i]
        im = axes[1, i].imshow(temp, cmap=THERMAL_CMAP)
        axes[1, i].set_title(f'{temp.max():.1f}°C max')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def create_sample_gallery(
    data_dir: str,
    num_samples: int = 9,
    output_path: Optional[str] = None
):
    """
    Create a gallery of samples from a generated dataset.
    
    Args:
        data_dir: Directory containing dataset files
        num_samples: Number of samples to show (will be made square)
        output_path: Optional path to save gallery
    """
    data_path = Path(data_dir)
    
    # Load training data
    train_data = np.load(data_path / 'train.npz')
    inputs = train_data['inputs']
    outputs = train_data['outputs']
    
    # Make grid square
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    num_samples = min(num_samples, len(inputs))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(3*grid_size, 3*grid_size))
    fig.suptitle('Dataset Sample Gallery (Temperature Fields)', fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            temp = outputs[idx, 0]
            im = ax.imshow(temp, cmap=THERMAL_CMAP)
            ax.set_title(f'Max: {temp.max():.1f}°C', fontsize=9)
        ax.axis('off')
        
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved gallery to {output_path}")
        
    return fig


if __name__ == "__main__":
    # Demo with random data
    print("Visualization demo - generating random sample...")
    
    h, w = 128, 128
    demo_inputs = np.random.rand(4, h, w).astype(np.float32)
    demo_output = 25 + 20 * np.random.rand(1, h, w).astype(np.float32)
    
    fig = plot_sample(demo_inputs, demo_output)
    plt.show()
