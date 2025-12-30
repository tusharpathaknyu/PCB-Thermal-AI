#!/usr/bin/env python3
"""
PCB Thermal AI - Training Script

Train U-Net models for PCB temperature prediction.

Usage:
    python scripts/train.py --data data/synthetic --epochs 50

Options:
    --data: Path to dataset directory
    --model: Model architecture (unet, unet_small, thermal_unet)
    --epochs: Number of training epochs
    --batch-size: Batch size
    --lr: Learning rate
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training import train_model, get_dataloaders
from models import get_model
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Train PCB thermal prediction model'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/synthetic',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='unet_small',
        choices=['unet', 'unet_small', 'thermal_unet'],
        help='Model architecture'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for tensorboard logs'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Check data exists
    if not Path(args.data).exists():
        print(f"Error: Data directory not found: {args.data}")
        print("Run 'python scripts/generate_dataset.py --quick' first")
        sys.exit(1)
        
    # Print configuration
    print("="*60)
    print("PCB Thermal AI - Training")
    print("="*60)
    print(f"Data:       {args.data}")
    print(f"Model:      {args.model}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Device:     {get_device()}")
    print("="*60)
    
    # Train
    trainer = train_model(
        data_path=args.data,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_workers=args.num_workers
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: {args.checkpoint_dir}/best.pth")
    print(f"Tensorboard logs:    {args.log_dir}/")
    print(f"\nTo view training curves:")
    print(f"  tensorboard --logdir {args.log_dir}")


def get_device() -> str:
    """Get available device"""
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    else:
        return "cpu"


if __name__ == "__main__":
    main()
