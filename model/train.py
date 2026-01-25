"""
Training script for phone-quality deepfake detector.

Features:
- TensorBoard logging for real-time monitoring
- Learning rate scheduling
- Model checkpointing
- Training history tracking
"""

import os
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import PhoneAudioDataset, create_dataloaders
from model import get_model, count_parameters, estimate_model_size
from phone_augment import PhoneConfig


class Trainer:
    """Training manager for phone deepfake detector."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        log_dir: str = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # TensorBoard logging
        if log_dir is None:
            log_dir = f"runs/deepfake_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        print(f"TensorBoard logs: {log_dir}")
        print(f"  Run: tensorboard --logdir {log_dir}")

    def train_epoch(self) -> tuple:
        """Train one epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate model."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(self.val_loader, desc="Validating"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.val_loader), 100. * correct / total

    def train(self, epochs: int, save_dir: str = "checkpoints"):
        """Full training loop with TensorBoard logging."""
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training Phone Deepfake Detector")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Model size: {estimate_model_size(self.model):.2f} MB")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        # Log model graph to TensorBoard (optional, can be slow)
        # sample_input = torch.randn(1, 1, 64, 94).to(self.device)
        # self.writer.add_graph(self.model, sample_input)

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # TensorBoard logging
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)

            # Log individual metrics too for easier comparison
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)

            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}\n")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"Saved best model (Val Acc: {val_acc:.2f}%)\n")

        # Save final checkpoint
        torch.save({
            'epoch': epochs,
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

        print(f"Training complete! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"View training logs: tensorboard --logdir {self.log_dir}")


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
                        help="TensorBoard log directory (default: runs/deepfake_<timestamp>)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        apply_degradation=not args.no_degrade
    )

    # Model
    model = get_model(args.model)

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_dir=args.log_dir
    )
    trainer.train(args.epochs, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
