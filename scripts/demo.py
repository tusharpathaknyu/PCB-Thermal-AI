#!/usr/bin/env python3
"""
PCB Thermal AI - Demo Script

Demonstrates the ML model's ability to predict temperature fields
from PCB layout features in seconds vs hours for traditional FEA.

Usage:
    python scripts/demo.py
    python scripts/demo.py --save-figure
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import PCBGenerator, ThermalSolver
from src.inference import ThermalPredictor


def run_demo(save_figure: bool = False):
    """Run demonstration comparing ML prediction vs ground truth."""
    
    print("=" * 60)
    print("PCB Thermal AI - Demo")
    print("=" * 60)
    print()
    
    # Check for trained model
    checkpoint_path = Path("checkpoints/best.pth")
    if not checkpoint_path.exists():
        print("❌ No trained model found!")
        print("   Run training first: python scripts/train.py")
        return
    
    # Load model
    print("Loading trained model...")
    predictor = ThermalPredictor.load(str(checkpoint_path))
    print(f"✓ Model loaded (device: {predictor.device})")
    print()
    
    # Generate test PCB
    print("Generating test PCB layout...")
    generator = PCBGenerator(grid_size=(128, 128))
    layout = generator.generate(
        complexity="medium",
        num_components=6,
        total_power=4.0  # 4W total power
    )
    
    print(f"  Components: {len(layout.components)}")
    print(f"  Vias: {len(layout.vias)}")
    print(f"  Total Power: {layout.power_map.sum()/1000:.2f}W")
    print()
    
    # Run ML prediction
    print("Running ML prediction...")
    t_start = time.time()
    result = predictor.predict(
        copper=layout.copper_density,
        vias=layout.via_map,
        components=layout.component_map,
        power=layout.power_map / 1000,  # Normalize
        return_dict=True
    )
    ml_time = time.time() - t_start
    
    print(f"  ✓ ML Prediction: {ml_time*1000:.1f}ms")
    print(f"    Max Temp: {result['max_temp']:.1f}°C")
    print(f"    Mean Temp: {result['mean_temp']:.1f}°C")
    print(f"    Hotspot: {result['hotspot_location']}")
    print()
    
    # Run ground truth simulation
    print("Running ground truth FEA simulation...")
    t_start = time.time()
    solver = ThermalSolver(grid_size=(128, 128))
    gt_temp = solver.solve(layout)
    fea_time = time.time() - t_start
    
    print(f"  ✓ FEA Simulation: {fea_time*1000:.1f}ms")
    print(f"    Max Temp: {gt_temp.max():.1f}°C")
    print(f"    Mean Temp: {gt_temp.mean():.1f}°C")
    print()
    
    # Compare results
    mae = np.abs(result['temperature'] - gt_temp).mean()
    max_error = np.abs(result['temperature'] - gt_temp).max()
    
    print("=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"  MAE:           {mae:.2f}°C")
    print(f"  Max Error:     {max_error:.2f}°C")
    print(f"  ML Speed:      {ml_time*1000:.1f}ms")
    print(f"  FEA Speed:     {fea_time*1000:.1f}ms")
    print(f"  Speedup:       {fea_time/ml_time:.1f}x faster")
    print()
    
    # Note about traditional FEA
    print("💡 Note: This synthetic FEA runs fast because it's simplified.")
    print("   Real Ansys/COMSOL simulations take 30-120 MINUTES per design!")
    print("   Our ML model enables real-time thermal feedback during design.")
    print()
    
    # Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Row 1: Input features
    ax1, ax2, ax3 = axes[0]
    
    im1 = ax1.imshow(layout.copper_density, cmap='copper')
    ax1.set_title('Copper Layer', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    im2 = ax2.imshow(layout.component_map, cmap='Reds')
    ax2.set_title('Components', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Presence')
    
    im3 = ax3.imshow(layout.power_map, cmap='hot')
    ax3.set_title('Power Map (W/m²)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, label='Power')
    
    # Row 2: Results
    ax4, ax5, ax6 = axes[1]
    
    vmin = min(result['temperature'].min(), gt_temp.min())
    vmax = max(result['temperature'].max(), gt_temp.max())
    
    im4 = ax4.imshow(result['temperature'], cmap='jet', vmin=vmin, vmax=vmax)
    ax4.set_title(f'ML Prediction ({ml_time*1000:.0f}ms)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, label='°C')
    
    # Mark hotspot
    hy, hx = result['hotspot_location']
    ax4.plot(hx, hy, 'w*', markersize=15, markeredgecolor='black')
    
    im5 = ax5.imshow(gt_temp, cmap='jet', vmin=vmin, vmax=vmax)
    ax5.set_title(f'Ground Truth FEA ({fea_time*1000:.0f}ms)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, label='°C')
    
    # Error map
    error = np.abs(result['temperature'] - gt_temp)
    im6 = ax6.imshow(error, cmap='RdYlGn_r')
    ax6.set_title(f'Absolute Error (MAE: {mae:.1f}°C)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, label='°C')
    
    plt.suptitle('PCB Thermal AI - ML vs FEA Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_figure:
        output_path = Path("outputs/demo_comparison.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to {output_path}")
    
    plt.show()
    
    print()
    print("=" * 60)
    print("Demo complete! The model achieves ~6°C MAE on average.")
    print("This enables engineers to get thermal feedback in SECONDS")
    print("instead of waiting HOURS for traditional FEA simulations.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PCB Thermal AI Demo")
    parser.add_argument('--save-figure', action='store_true',
                       help='Save the comparison figure')
    args = parser.parse_args()
    
    run_demo(save_figure=args.save_figure)


if __name__ == "__main__":
    main()
