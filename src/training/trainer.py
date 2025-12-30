"""
Training Pipeline for PCB Thermal Predictor

Handles:
- Model training loop
- Validation
- Checkpointing
- Logging
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model
from training.dataset import get_dataloaders


class ThermalLoss(nn.Module):
    """
    Combined loss for thermal prediction.
    
    Components:
    - MSE: Mean squared error for overall accuracy
    - Hotspot loss: Extra penalty for max temperature errors
    - Gradient loss: Preserve thermal gradients
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        hotspot_weight: float = 0.2,
        gradient_weight: float = 0.1
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.hotspot_weight = hotspot_weight
        self.gradient_weight = gradient_weight
        self.mse = nn.MSELoss()
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted temperature (B, 1, H, W)
            target: Target temperature (B, 1, H, W)
            
        Returns:
            total_loss, loss_components
        """
        # MSE loss
        mse_loss = self.mse(pred, target)
        
        # Hotspot loss (max temperature per sample)
        pred_max = pred.view(pred.size(0), -1).max(dim=1)[0]
        target_max = target.view(target.size(0), -1).max(dim=1)[0]
        hotspot_loss = self.mse(pred_max, target_max)
        
        # Gradient loss (preserve spatial gradients)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        gradient_loss = (
            self.mse(pred_dx, target_dx) + 
            self.mse(pred_dy, target_dy)
        ) / 2
        
        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.hotspot_weight * hotspot_loss +
            self.gradient_weight * gradient_loss
        )
        
        loss_components = {
            'mse': mse_loss.item(),
            'hotspot': hotspot_loss.item(),
            'gradient': gradient_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components


class Trainer:
    """
    Training manager for PCB thermal prediction models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_stats: Dict[str, float],
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_stats = output_stats
        self.config = config
        
        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Loss
        self.criterion = ThermalLoss(
            mse_weight=config.get('mse_weight', 1.0),
            hotspot_weight=config.get('hotspot_weight', 0.2),
            gradient_weight=config.get('gradient_weight', 0.1)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 10)
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss, loss_components = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            epoch_losses.append(loss_components)
            pbar.set_postfix({'loss': f"{loss_components['total']:.4f}"})
            
        # Average losses
        avg_losses = {
            k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0].keys()
        }
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        epoch_losses = []
        
        # For metrics in original scale
        all_mae = []
        all_max_errors = []
        
        for inputs, targets in tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss, loss_components = self.criterion(outputs, targets)
            epoch_losses.append(loss_components)
            
            # Denormalize for real metrics
            outputs_denorm = outputs * self.output_stats['std'] + self.output_stats['mean']
            targets_denorm = targets * self.output_stats['std'] + self.output_stats['mean']
            
            # MAE in °C
            mae = torch.abs(outputs_denorm - targets_denorm).mean().item()
            all_mae.append(mae)
            
            # Max temperature error
            pred_max = outputs_denorm.view(outputs.size(0), -1).max(dim=1)[0]
            target_max = targets_denorm.view(targets.size(0), -1).max(dim=1)[0]
            max_error = torch.abs(pred_max - target_max).mean().item()
            all_max_errors.append(max_error)
            
        avg_losses = {
            k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0].keys()
        }
        
        metrics = {
            'mae_celsius': sum(all_mae) / len(all_mae),
            'max_temp_error': sum(all_max_errors) / len(all_max_errors)
        }
        
        return avg_losses, metrics
    
    def train(self, num_epochs: int):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses, val_metrics = self.validate()
            self.val_losses.append(val_losses)
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Logging
            self.writer.add_scalars('Loss/train', train_losses, epoch)
            self.writer.add_scalars('Loss/val', val_losses, epoch)
            self.writer.add_scalars('Metrics/val', val_metrics, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss:   {val_losses['total']:.4f}")
            print(f"  MAE:        {val_metrics['mae_celsius']:.2f}°C")
            print(f"  Max Error:  {val_metrics['max_temp_error']:.2f}°C")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best.pth')
                print(f"  ✓ New best model saved!")
                
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pth')
                
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint('final.pth')
        
        # Save training history
        self.save_history()
        
        self.writer.close()
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'output_stats': self.output_stats,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {self.epoch}")
        
    def save_history(self):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)


def train_model(
    data_path: str,
    model_name: str = "unet_small",
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    **kwargs
) -> Trainer:
    """
    Convenience function to train a model.
    
    Args:
        data_path: Path to dataset
        model_name: Model architecture name
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        **kwargs: Additional config options
        
    Returns:
        Trained Trainer instance
    """
    # Create dataloaders
    train_loader, val_loader, test_loader, output_stats = get_dataloaders(
        data_path,
        batch_size=batch_size,
        num_workers=kwargs.get('num_workers', 4)
    )
    
    # Create model
    model = get_model(model_name)
    
    # Config
    config = {
        'model_name': model_name,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        **kwargs
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_stats=output_stats,
        config=config
    )
    
    # Train
    trainer.train(num_epochs)
    
    return trainer


if __name__ == "__main__":
    # Quick test
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "data" / "synthetic"
    
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        print("Run 'python scripts/generate_dataset.py --quick' first")
    else:
        trainer = train_model(
            str(data_path),
            model_name="unet_small",
            num_epochs=5,
            batch_size=8,
            num_workers=0
        )
