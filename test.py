import os
import json
import random
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from load_data import get_data_loaders
from models import MODEL_REGISTRY
from utils import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on test set and plot confusion matrix')
    parser.add_argument('--model', type=str, default='baseline', choices=sorted(MODEL_REGISTRY.keys()),
                        help='Model name to evaluate')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    _, _, test_loader, classes = get_data_loaders(
        train_dir=os.path.join(base_dir, 'train_images'),
        test_dir=os.path.join(base_dir, 'test_images'),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model,
        use_augmentation=False,
    )

    try:
        model = load_model_from_checkpoint(args.model, device, base_dir=base_dir)
    except FileNotFoundError as e:
        print(f"Checkpoint not found. Train the selected model first.")
        print(f"python {os.path.join(base_dir, 'train.py')} --model {args.model}")
        return

    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    print(f"Correct: {correct} of {total} ({acc*100:.2f}%)")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(classes))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    artifacts_dir = os.path.join(base_dir, 'artifacts', args.model)
    os.makedirs(artifacts_dir, exist_ok=True)
    plt.savefig(os.path.join(artifacts_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    samples_to_show = 10
    shown = 0
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes = axes.ravel()

    for images, labels in test_loader:
        with torch.no_grad():
            outputs = model(images.to(device))
            probs = F.softmax(outputs, dim=1).cpu()
            preds = probs.argmax(dim=1)
        for i in range(images.size(0)):
            if shown >= samples_to_show:
                break
            img_t = images[i]
            if img_t.shape[0] == 1:
                img = img_t.squeeze(0).numpy() * 0.5 + 0.5
                axes[shown].imshow(img, cmap='gray')
            else:
                img = img_t.clone().cpu().numpy()
                img[0] = img[0] * 0.229 + 0.485
                img[1] = img[1] * 0.224 + 0.456
                img[2] = img[2] * 0.225 + 0.406
                img = img.transpose(1, 2, 0).clip(0, 1)
                axes[shown].imshow(img)
            pred_cls = classes[preds[i]]
            true_cls = classes[labels[i]]
            conf = probs[i, preds[i]].item()
            axes[shown].set_title(f"pred: {pred_cls} ({conf:.2f})\ntrue: {true_cls}")
            axes[shown].axis('off')
            shown += 1
        if shown >= samples_to_show:
            break

    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'sample_predictions.png'), bbox_inches='tight')
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


if __name__ == '__main__':
    main()
