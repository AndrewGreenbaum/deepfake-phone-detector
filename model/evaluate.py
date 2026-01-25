"""
Comprehensive Model Evaluation Script

Computes full metrics for deepfake detection model:
- Accuracy, Precision, Recall, F1-score
- AUC-ROC and precision-recall curves
- Confusion matrix visualization
- Per-class metrics breakdown
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model import get_model
from dataset import create_dataloaders, PhoneAudioDataset, PreDegradedDataset
from phone_augment import PhoneConfig


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        output_dir: str = "evaluation_results"
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(device)
        self.model.eval()

    def evaluate(self, dataloader) -> dict:
        """Run evaluation on dataloader and compute all metrics."""
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = batch
                inputs = inputs.to(self.device)

                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake

        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, y_prob)

        # Generate visualizations
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_precision_recall_curve(y_true, y_prob)

        # Save metrics to JSON
        self._save_metrics(metrics)

        return metrics

    def _compute_metrics(self, y_true, y_pred, y_prob) -> dict:
        """Compute all classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
            "total_samples": len(y_true),
            "real_samples": int((y_true == 0).sum()),
            "fake_samples": int((y_true == 1).sum()),
            "predicted_real": int((y_pred == 0).sum()),
            "predicted_fake": int((y_pred == 1).sum()),
        }

        # Per-class metrics
        report = classification_report(y_true, y_pred, target_names=["REAL", "FAKE"], output_dict=True)
        metrics["per_class"] = {
            "real": {
                "precision": report["REAL"]["precision"],
                "recall": report["REAL"]["recall"],
                "f1": report["REAL"]["f1-score"],
                "support": report["REAL"]["support"]
            },
            "fake": {
                "precision": report["FAKE"]["precision"],
                "recall": report["FAKE"]["recall"],
                "f1": report["FAKE"]["f1-score"],
                "support": report["FAKE"]["support"]
            }
        }

        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Generate and save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Deepfake Detection')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        print(f"Saved confusion matrix to {self.output_dir / 'confusion_matrix.png'}")

    def _plot_roc_curve(self, y_true, y_prob):
        """Generate and save ROC curve plot."""
        if len(np.unique(y_true)) < 2:
            print("Warning: Only one class in test set, skipping ROC curve")
            return

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Deepfake Detection')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=150)
        plt.close()
        print(f"Saved ROC curve to {self.output_dir / 'roc_curve.png'}")

    def _plot_precision_recall_curve(self, y_true, y_prob):
        """Generate and save precision-recall curve plot."""
        if len(np.unique(y_true)) < 2:
            print("Warning: Only one class in test set, skipping PR curve")
            return

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'g-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Deepfake Detection')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=150)
        plt.close()
        print(f"Saved precision-recall curve to {self.output_dir / 'precision_recall_curve.png'}")

    def _save_metrics(self, metrics: dict):
        """Save metrics to JSON file."""
        metrics["timestamp"] = datetime.now().isoformat()

        output_path = self.output_dir / 'metrics.json'
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {output_path}")


def print_metrics(metrics: dict):
    """Pretty print evaluation metrics."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    print(f"\nDataset Statistics:")
    print(f"  Total samples:    {metrics['total_samples']}")
    print(f"  Real samples:     {metrics['real_samples']}")
    print(f"  Fake samples:     {metrics['fake_samples']}")
    print(f"  Predicted real:   {metrics['predicted_real']}")
    print(f"  Predicted fake:   {metrics['predicted_fake']}")

    print(f"\nPer-Class Metrics:")
    for cls_name, cls_metrics in metrics['per_class'].items():
        print(f"  {cls_name.upper()}:")
        print(f"    Precision: {cls_metrics['precision']:.4f}")
        print(f"    Recall:    {cls_metrics['recall']:.4f}")
        print(f"    F1:        {cls_metrics['f1']:.4f}")
        print(f"    Support:   {cls_metrics['support']}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="mobile",
                        choices=["phone_cnn", "temporal", "mobile", "attention"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which data split to evaluate")
    parser.add_argument("--pre-degraded", action="store_true",
                        help="Use pre-degraded phone audio dataset")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = get_model(args.model)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_accuracy' in checkpoint:
        print(f"  Val Accuracy: {checkpoint['val_accuracy']:.4f}")

    # Create dataloader
    print(f"\nLoading {args.split} dataset from {args.data}")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        pre_degraded=args.pre_degraded
    )

    if args.split == "train":
        dataloader = train_loader
    elif args.split == "val":
        dataloader = val_loader
    else:
        dataloader = test_loader

    print(f"  Samples: {len(dataloader.dataset)}")

    # Run evaluation
    evaluator = ModelEvaluator(model, args.device, args.output)
    metrics = evaluator.evaluate(dataloader)

    # Print results
    print_metrics(metrics)

    print(f"\nResults saved to: {args.output}/")
    print("  - metrics.json")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")


if __name__ == "__main__":
    main()
