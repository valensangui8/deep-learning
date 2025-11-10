import os
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import seaborn as sns
import numpy as np

from models import MODEL_REGISTRY
from load_data import get_data_loaders


def load_checkpoint(model_name: str, device: torch.device):
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'artifacts', model_name, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado para {model_name}: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    Model = MODEL_REGISTRY[model_name]
    model = Model().to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, checkpoint_path


def evaluate_on_loader(model, dataloader, device: torch.device) -> Tuple[List[int], List[int]]:
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())
    return all_labels, all_preds


def plot_bar(df: pd.DataFrame, metric: str, out_path: str):
    plt.figure(figsize=(10, 5))
    sns.barplot(x='model', y=metric, data=df, palette='Blues_d')
    plt.ylabel(metric)
    plt.xlabel('model')
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_confusion(cm, classes: Tuple[str, str], out_path: str, title: str):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_bootstrap: int, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    accs, precs, recs, f1s = [], [], [], []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', pos_label=1, zero_division=0)
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
    return {
        'accuracy': np.array(accs),
        'precision': np.array(precs),
        'recall': np.array(recs),
        'f1': np.array(f1s),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluar y comparar métricas de todos los modelos')
    parser.add_argument('--models', type=str, default='all', help="Lista separada por comas o 'all'")
    parser.add_argument('--save-dir', type=str, default=None, help='Directorio de salida para resultados/plots')
    parser.add_argument('--bootstrap', type=int, default=0, help='Número de remuestreos bootstrap (0 para desactivar)')
    parser.add_argument('--ci', type=float, default=95.0, help='Intervalo de confianza en %% (ej. 95)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_dir = os.path.dirname(__file__)
    out_dir = args.save_dir or os.path.join(project_dir, 'artifacts', 'evaluation')
    os.makedirs(out_dir, exist_ok=True)

    available = sorted(MODEL_REGISTRY.keys())
    if args.models.lower() == 'all':
        models_to_eval = available
    else:
        requested = [m.strip() for m in args.models.split(',') if m.strip()]
        unknown = [m for m in requested if m not in available]
        if unknown:
            raise ValueError(f"Modelos desconocidos: {unknown}\nDisponibles: {available}")
        models_to_eval = requested

    rows: List[Dict] = []
    classes = ('noface', 'face')
    per_model_cm = {}
    bootstrap_rows: List[Dict] = []

    for idx, model_name in enumerate(models_to_eval, 1):
        print(f"[{idx}/{len(models_to_eval)}] Evaluando {model_name}...")
        # Get dataloaders with correct preprocessing for this model
        _, _, test_loader, _ = get_data_loaders(
            train_dir=os.path.join(project_dir, 'train_images'),
            test_dir=os.path.join(project_dir, 'test_images'),
            batch_size=64,
            valid_size=0.2,
            num_workers=2,
            shuffle_dataset=False,
            model_name=model_name,
            use_augmentation=False,
        )
        try:
            model, ckpt_path = load_checkpoint(model_name, device)
        except FileNotFoundError as e:
            print(f"⚠️  {e}. Omitiendo...")
            continue

        y_true, y_pred = evaluate_on_loader(model, test_loader, device)
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        acc = accuracy_score(y_true_np, y_pred_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', pos_label=1, zero_division=0
        )
        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
        per_model_cm[model_name] = cm

        row = {
            'model': model_name,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'checkpoint': ckpt_path,
        }
        rows.append(row)
        print(f"   acc={acc:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}")

        # Bootstrap (optional)
        if args.bootstrap and len(y_true_np) > 0:
            boot = bootstrap_metrics(y_true_np, y_pred_np, args.bootstrap, random_state=42)
            alpha = (100 - args.ci) / 2.0
            low_q, high_q = alpha, 100 - alpha
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                arr = boot[metric]
                bootstrap_rows.append({
                    'model': model_name,
                    'metric': metric,
                    'mean': float(arr.mean()),
                    'std': float(arr.std(ddof=1)),
                    'ci_low': float(np.percentile(arr, low_q)),
                    'ci_high': float(np.percentile(arr, high_q)),
                })
            # Save per-model distributions
            boot_df = pd.DataFrame({
                'accuracy': boot['accuracy'],
                'precision': boot['precision'],
                'recall': boot['recall'],
                'f1': boot['f1'],
            })
            boot_df.to_csv(os.path.join(out_dir, f'bootstrap_{model_name}.csv'), index=False)

    if not rows:
        print("No se evaluó ningún modelo. Revisa que existan checkpoints en artifacts/<modelo>/best_model.pt")
        return

    df = pd.DataFrame(rows).sort_values(by='accuracy', ascending=False)
    csv_path = os.path.join(out_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResumen guardado en: {csv_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plots
    plot_bar(df, 'accuracy', os.path.join(out_dir, 'accuracy.png'))
    plot_bar(df, 'f1', os.path.join(out_dir, 'f1.png'))
    plot_bar(df, 'precision', os.path.join(out_dir, 'precision.png'))
    plot_bar(df, 'recall', os.path.join(out_dir, 'recall.png'))
    print(f"Gráficos guardados en: {out_dir}")

    # Confusion matrices per top-3 models by acc
    top_models = df['model'].head(min(3, len(df))).tolist()
    for m in top_models:
        cm = per_model_cm[m]
        out = os.path.join(out_dir, f'cm_{m}.png')
        plot_confusion(cm, classes, out, title=f'Confusion Matrix - {m}')

    # Bootstrap summary and violin plots (if requested)
    if bootstrap_rows:
        boot_summary = pd.DataFrame(bootstrap_rows)
        boot_csv = os.path.join(out_dir, 'bootstrap_summary.csv')
        boot_summary.to_csv(boot_csv, index=False)
        print(f"Resumen bootstrap guardado en: {boot_csv}")

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plt.figure(figsize=(10, 5))
            parts = []
            for m in df['model'].tolist():
                path = os.path.join(out_dir, f'bootstrap_{m}.csv')
                if os.path.exists(path):
                    bdf = pd.read_csv(path)
                    parts.append(pd.DataFrame({'model': m, metric: bdf[metric]}))
            if parts:
                cat_df = pd.concat(parts, axis=0, ignore_index=True)
                sns.violinplot(x='model', y=metric, data=cat_df, palette='Pastel1', cut=0, inner='quartile')
                plt.ylabel(metric)
                plt.xlabel('model')
                plt.ylim(0.0, 1.0)
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'{metric}_violin.png'), bbox_inches='tight')
                plt.close()


if __name__ == '__main__':
    main()


