#!/usr/bin/env python3
"""
PCB Thermal AI - Dataset Generation Script

This script generates synthetic training data for the PCB thermal predictor.
Run this to create your initial dataset before training.

Usage:
    python scripts/generate_dataset.py --num-samples 1000 --output data/synthetic

Options:
    --num-samples: Number of samples to generate (default: 1000)
    --output: Output directory (default: data/synthetic)
    --grid-size: Grid resolution (default: 128)
    --visualize: Generate sample visualizations
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_generation import DatasetGenerator, DatasetConfig, quick_generate
from data_generation.visualize import (
    plot_sample, 
    create_sample_gallery,
    plot_dataset_statistics
)
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic PCB thermal training data'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=1000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/synthetic',
        help='Output directory'
    )
    parser.add_argument(
        '--grid-size', '-g',
        type=int,
        default=128,
        help='Grid resolution (square)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization samples'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode - generate only 100 samples for testing'
    )
    
    args = parser.parse_args()
    
    # Quick mode for testing
    if args.quick:
        print("Quick mode: generating 100 samples...")
        stats = quick_generate(num_samples=100, output_dir=args.output)
        
    else:
        # Full generation
        print(f"Generating {args.num_samples} samples at {args.grid_size}x{args.grid_size} resolution...")
        
        config = DatasetConfig(
            grid_size=(args.grid_size, args.grid_size),
            num_samples=args.num_samples,
            output_format='npz'
        )
        
        generator = DatasetGenerator(config)
        stats = generator.generate_dataset(args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Output directory: {args.output}")
    print(f"Total samples: {stats['num_samples']}")
    print(f"  - Train: {stats['splits']['train']}")
    print(f"  - Val: {stats['splits']['val']}")
    print(f"  - Test: {stats['splits']['test']}")
    print(f"\nTemperature range: {stats['temperature']['min']:.1f}°C - {stats['temperature']['max']:.1f}°C")
    print(f"Power range: {stats['power']['min']:.2f}W - {stats['power']['max']:.2f}W")
    print(f"Mean max temperature: {stats['max_temps']['mean']:.1f}°C")
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        output_path = Path(args.output)
        
        # Sample gallery
        create_sample_gallery(
            args.output,
            num_samples=9,
            output_path=output_path / 'sample_gallery.png'
        )
        
        # Statistics plot
        plot_dataset_statistics(
            output_path / 'dataset_stats.json',
            save_path=output_path / 'statistics.png'
        )
        
        # Individual sample visualization
        train_data = np.load(output_path / 'train.npz')
        
        # Load metadata for the first sample
        import json
        with open(output_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        plot_sample(
            train_data['inputs'][0],
            train_data['outputs'][0],
            metadata=metadata[0] if metadata else None,
            save_path=output_path / 'sample_detailed.png'
        )
        
        print(f"Visualizations saved to {output_path}")
        
    print("\nDone! Your dataset is ready for training.")
    print(f"\nNext steps:")
    print(f"  1. Review samples in {args.output}/")
    print(f"  2. Implement U-Net model in src/models/")
    print(f"  3. Train: python scripts/train.py --data {args.output}")


if __name__ == "__main__":
    main()
