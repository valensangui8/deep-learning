"""
Advanced model comparison analysis with additional visualizations
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from models import MODEL_REGISTRY

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_training_history(model_name: str) -> Dict:
    """Load training history JSON"""
    history_path = Path("artifacts") / model_name / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None


def load_summary() -> pd.DataFrame:
    """Load evaluation summary CSV"""
    summary_path = Path("artifacts/evaluation/summary.csv")
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return None


def plot_training_curves(df_summary: pd.DataFrame, out_dir: Path):
    """Plot training curves (loss and accuracy) for all models"""
    models_with_history = []
    histories = {}
    
    for model_name in df_summary['model'].tolist():
        hist = load_training_history(model_name)
        if hist:
            histories[model_name] = hist
            models_with_history.append(model_name)
    
    if not histories:
        print("No training histories found. Skipping training curves.")
        return
    
    # Plot loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name in models_with_history:
        hist = histories[model_name]
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], label=f"{model_name} (train)", linestyle='-', alpha=0.7)
        axes[0].plot(epochs, hist['valid_loss'], label=f"{model_name} (valid)", linestyle='--', alpha=0.7)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy curves
    for model_name in models_with_history:
        hist = histories[model_name]
        epochs = range(1, len(hist['train_acc']) + 1)
        axes[1].plot(epochs, hist['train_acc'], label=f"{model_name} (train)", linestyle='-', alpha=0.7)
        axes[1].plot(epochs, hist['valid_acc'], label=f"{model_name} (valid)", linestyle='--', alpha=0.7)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Training curves saved to {out_dir / 'training_curves.png'}")


def plot_metrics_comparison(df_summary: pd.DataFrame, out_dir: Path):
    """Create a comprehensive metrics comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    df_sorted = df_summary.sort_values('accuracy', ascending=True)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        colors = ['#3498db' if 'pretrained' in m or m in ['resnet18', 'mobilenetv2', 'efficientnet'] 
                  else '#2ecc71' for m in df_sorted['model']]
        
        bars = ax.barh(df_sorted['model'], df_sorted[metric], color=colors, alpha=0.8)
        ax.set_xlabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'metrics_comparison_comprehensive.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Comprehensive metrics comparison saved to {out_dir / 'metrics_comparison_comprehensive.png'}")


def plot_model_parameters(df_summary: pd.DataFrame, out_dir: Path):
    """Plot model size (parameters) vs performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    param_counts = {}
    
    for model_name in df_summary['model'].tolist():
        try:
            # Try to load from checkpoint first (avoids downloading pretrained weights)
            checkpoint_path = Path("artifacts") / model_name / "best_model.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                ModelClass = MODEL_REGISTRY[model_name]
                model = ModelClass().to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                param_counts[model_name] = count_parameters(model)
            else:
                # Fallback: create new instance (may download for pretrained)
                ModelClass = MODEL_REGISTRY[model_name]
                model = ModelClass().to(device)
                param_counts[model_name] = count_parameters(model)
        except Exception as e:
            print(f"  ⚠️  Could not count parameters for {model_name}: {e}")
            param_counts[model_name] = None
    
    df_params = df_summary.copy()
    df_params['parameters'] = df_params['model'].map(param_counts)
    df_params = df_params.dropna(subset=['parameters'])
    
    if len(df_params) == 0:
        print("No parameter counts available. Skipping parameter analysis.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics_to_plot = ['accuracy', 'f1', 'precision']
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        scatter = ax.scatter(df_params['parameters'], df_params[metric], 
                           s=100, alpha=0.6, c=range(len(df_params)), cmap='viridis')
        
        # Add labels
        for _, row in df_params.iterrows():
            ax.annotate(row['model'], (row['parameters'], row[metric]), 
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Number of Parameters (log scale)')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} vs Model Size')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'parameters_vs_performance.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Parameters vs performance saved to {out_dir / 'parameters_vs_performance.png'}")
    
    # Save parameter table
    param_df = df_params[['model', 'parameters', 'accuracy', 'precision', 'recall', 'f1']].copy()
    param_df = param_df.sort_values('parameters')
    param_df['parameters'] = param_df['parameters'].apply(lambda x: f"{x:,}")
    param_df.to_csv(out_dir / 'model_parameters.csv', index=False)
    print(f"Parameter table saved to {out_dir / 'model_parameters.csv'}")


def plot_confusion_matrices_comparison(df_summary: pd.DataFrame, out_dir: Path):
    """Create a grid of confusion matrices for top models"""
    from evaluate_models import load_checkpoint, evaluate_on_loader
    from load_data import get_data_loaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_dir = Path(__file__).parent
    
    # Get top 6 models by accuracy, but only those with checkpoints
    top_models = df_summary.nlargest(6, 'accuracy')['model'].tolist()
    available_models = []
    for m in top_models:
        checkpoint_path = Path("artifacts") / m / "best_model.pt"
        if checkpoint_path.exists():
            available_models.append(m)
        else:
            print(f"  ⚠️  Skipping {m}: checkpoint not found")
    
    if len(available_models) == 0:
        print("  ⚠️  No models with checkpoints found. Skipping confusion matrices grid.")
        return
    
    # Adjust grid size based on available models
    n_models = len(available_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(available_models):
        try:
            print(f"    Processing {model_name}...")
            _, _, test_loader, _ = get_data_loaders(
                train_dir=project_dir / 'train_images',
                test_dir=project_dir / 'test_images',
                batch_size=64,
                valid_size=0.2,
                num_workers=0,  # Avoid shm issues
                shuffle_dataset=False,
                model_name=model_name,
                use_augmentation=False,
            )
            model, _ = load_checkpoint(model_name, device)
            y_true, y_pred = evaluate_on_loader(model, test_loader, device)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                       xticklabels=['no-face', 'face'], yticklabels=['no-face', 'face'])
            acc = df_summary[df_summary["model"]==model_name]["accuracy"].values[0]
            ax.set_title(f'{model_name}\nAcc: {acc:.3f}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        except Exception as e:
            print(f"  ⚠️  Could not generate confusion matrix for {model_name}: {e}")
            axes[idx].axis('off')
            axes[idx].text(0.5, 0.5, f'{model_name}\nError', ha='center', va='center')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrices_grid.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Confusion matrices grid saved to {out_dir / 'confusion_matrices_grid.png'}")


def plot_metric_correlation(df_summary: pd.DataFrame, out_dir: Path):
    """Plot correlation heatmap between metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    corr_matrix = df_summary[metrics].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Metric Correlation Matrix')
    plt.tight_layout()
    plt.savefig(out_dir / 'metric_correlation.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Metric correlation saved to {out_dir / 'metric_correlation.png'}")


def create_detailed_comparison_table(df_summary: pd.DataFrame, out_dir: Path):
    """Create a detailed comparison table with rankings"""
    df_ranked = df_summary.copy()
    
    # Add rankings
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        df_ranked[f'{metric}_rank'] = df_ranked[metric].rank(ascending=False, method='min').astype(int)
    
    # Calculate average rank
    rank_cols = [col for col in df_ranked.columns if col.endswith('_rank')]
    df_ranked['avg_rank'] = df_ranked[rank_cols].mean(axis=1)
    df_ranked = df_ranked.sort_values('avg_rank')
    
    # Format for display
    display_cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'avg_rank']
    df_display = df_ranked[display_cols].copy()
    
    # Format numbers
    for col in ['accuracy', 'precision', 'recall', 'f1']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
    df_display['avg_rank'] = df_display['avg_rank'].apply(lambda x: f"{x:.2f}")
    
    df_display.to_csv(out_dir / 'detailed_comparison_table.csv', index=False)
    print(f"Detailed comparison table saved to {out_dir / 'detailed_comparison_table.csv'}")
    
    # Also save with rankings
    rank_display_cols = ['model'] + rank_cols + ['avg_rank']
    df_rank_display = df_ranked[rank_display_cols].copy()
    df_rank_display = df_rank_display.sort_values('avg_rank')
    df_rank_display.to_csv(out_dir / 'model_rankings.csv', index=False)
    print(f"Model rankings saved to {out_dir / 'model_rankings.csv'}")


def plot_precision_recall_tradeoff(df_summary: pd.DataFrame, out_dir: Path):
    """Plot precision vs recall to show trade-offs"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by model type
    colors = []
    for model in df_summary['model']:
        if model in ['resnet18', 'mobilenetv2', 'efficientnet']:
            colors.append('#e74c3c')  # Red for pretrained
        else:
            colors.append('#3498db')  # Blue for custom
    
    scatter = ax.scatter(df_summary['recall'], df_summary['precision'], 
                        s=200, alpha=0.6, c=colors, edgecolors='black', linewidth=1.5)
    
    # Add labels
    for _, row in df_summary.iterrows():
        ax.annotate(row['model'], (row['recall'], row['precision']), 
                   fontsize=9, alpha=0.8, ha='center')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add diagonal (F1 isolines)
    f1_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    for f1 in f1_levels:
        x = np.linspace(0, 1, 100)
        y = f1 * x / (2 * x - f1)
        y = np.clip(y, 0, 1)
        ax.plot(x, y, '--', alpha=0.2, color='gray', linewidth=1)
        # Add F1 label
        if f1 >= 0.7:
            ax.text(0.95, f1 * 0.95 / (2 * 0.95 - f1), f'F1={f1}', 
                   fontsize=8, alpha=0.5, ha='right')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Pretrained Models'),
        Patch(facecolor='#3498db', label='Custom CNNs')
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'precision_recall_tradeoff.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Precision-recall trade-off saved to {out_dir / 'precision_recall_tradeoff.png'}")


def main():
    parser = argparse.ArgumentParser(description='Advanced model analysis and visualization')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: artifacts/analysis)')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir) if args.output_dir else Path('artifacts/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summary
    df_summary = load_summary()
    if df_summary is None:
        print("Error: Could not load artifacts/evaluation/summary.csv")
        print("Please run evaluate_models.py first.")
        return
    
    print(f"Analyzing {len(df_summary)} models...")
    
    # Generate all analyses
    print("\n1. Creating comprehensive metrics comparison...")
    plot_metrics_comparison(df_summary, out_dir)
    
    print("\n2. Analyzing model parameters vs performance...")
    plot_model_parameters(df_summary, out_dir)
    
    print("\n3. Plotting training curves...")
    plot_training_curves(df_summary, out_dir)
    
    print("\n4. Creating confusion matrices grid...")
    try:
        plot_confusion_matrices_comparison(df_summary, out_dir)
    except KeyboardInterrupt:
        print("  ⚠️  Interrupted by user. Skipping confusion matrices.")
    except Exception as e:
        print(f"  ⚠️  Error generating confusion matrices: {e}")
    
    print("\n5. Computing metric correlations...")
    plot_metric_correlation(df_summary, out_dir)
    
    print("\n6. Creating precision-recall trade-off plot...")
    plot_precision_recall_tradeoff(df_summary, out_dir)
    
    print("\n7. Generating detailed comparison tables...")
    create_detailed_comparison_table(df_summary, out_dir)
    
    print(f"\n✅ All analyses complete! Results saved to {out_dir}")
    print("\nGenerated files:")
    for f in sorted(out_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()

