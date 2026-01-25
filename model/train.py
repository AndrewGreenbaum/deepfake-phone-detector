"""
Training script for phone-quality deepfake detector.

Features:
- TensorBoard logging for real-time monitoring
- Learning rate scheduling with warmup
- Model checkpointing
- Training history tracking
- Early stopping
- Mixed precision training (AMP)
- Gradient accumulation
- Class imbalance handling
- Reproducible training with seeds
"""

import os
import argparse
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import PhoneAudioDataset, create_dataloaders
from model import get_model, count_parameters, estimate_model_size
from phone_augment import PhoneConfig


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def compute_class_weights(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    class_counts = torch.zeros(2)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (2 * class_counts)

    print(f"Class distribution: Real={int(class_counts[0])}, Fake={int(class_counts[1])}")
    print(f"Class weights: Real={weights[0]:.3f}, Fake={weights[1]:.3f}")

    return weights.to(device)


class Trainer:
    """Training manager for phone deepfake detector with all best practices."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        log_dir: str = None,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        warmup_epochs: int = 5,
        class_weights: torch.Tensor = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.warmup_epochs = warmup_epochs

        # Loss with optional class weighting
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using weighted loss for class imbalance")
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate scheduler with warmup
        self.base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        # Mixed precision training
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("Mixed precision training enabled (AMP)")

        # TensorBoard logging
        if log_dir is None:
            log_dir = f"runs/deepfake_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        print(f"TensorBoard logs: {log_dir}")
        print(f"  Run: tensorboard --logdir {log_dir}")

    def _get_lr_scale(self, epoch: int) -> float:
        """Get learning rate scale for warmup."""
        if epoch <= self.warmup_epochs:
            return epoch / self.warmup_epochs
        return 1.0

    def train_epoch(self, epoch: int) -> tuple:
        """Train one epoch with AMP and gradient accumulation."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        # Learning rate warmup
        lr_scale = self._get_lr_scale(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale if epoch <= self.warmup_epochs else param_group['lr']

        pbar = tqdm(self.train_loader, desc=f"Training (E{epoch})")
        self.optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels) / self.grad_accum_steps

                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels) / self.grad_accum_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item() * self.grad_accum_steps:.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate model with optional AMP."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(self.val_loader, desc="Validating"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.val_loader), 100. * correct / total

    def train(
        self,
        epochs: int,
        save_dir: str = "checkpoints",
        patience: int = 10
    ) -> bool:
        """
        Full training loop with early stopping.

        Args:
            epochs: Maximum number of epochs
            save_dir: Checkpoint directory
            patience: Early stopping patience (epochs without improvement)

        Returns:
            True if training completed, False if early stopped
        """
        os.makedirs(save_dir, exist_ok=True)

        # Store initial LR for warmup
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        print(f"\n{'='*60}")
        print(f"Training Phone Deepfake Detector")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Model size: {estimate_model_size(self.model):.2f} MB")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.grad_accum_steps}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"{'='*60}\n")

        epochs_without_improvement = 0
        early_stopped = False

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Update scheduler after warmup
            if epoch > self.warmup_epochs:
                self.base_scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # TensorBoard logging
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)

            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")

            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"** New best model saved (Val Acc: {val_acc:.2f}%) **")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

            print()

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                early_stopped = True
                break

        # Save final checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, os.path.join(save_dir, 'final_model.pt'))

        # Save history
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        # Close TensorBoard writer
        self.writer.close()

        status = "early stopped" if early_stopped else "completed"
        print(f"Training {status}! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"View training logs: tensorboard --logdir {self.log_dir}")

        return not early_stopped


def main():
    parser = argparse.ArgumentParser(description="Train phone deepfake detector")
    parser.add_argument("--data", type=str, default="../data", help="Data path")
    parser.add_argument("--model", type=str, default="mobile",
                        choices=["phone_cnn", "temporal", "mobile", "attention"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--no-degrade", action="store_true", help="Skip phone degradation")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="TensorBoard log directory")

    # New arguments for best practices
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class weighting for imbalanced data")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        apply_degradation=not args.no_degrade
    )

    # Compute class weights if enabled
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_loader, device)

    # Model
    model = get_model(args.model)

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_dir=args.log_dir,
        use_amp=not args.no_amp,
        grad_accum_steps=args.grad_accum,
        warmup_epochs=args.warmup_epochs,
        class_weights=class_weights
    )
    trainer.train(args.epochs, save_dir=args.save_dir, patience=args.patience)


if __name__ == "__main__":
    main()
