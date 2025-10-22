import os
import json
import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from load_data import get_data_loaders
from net import Net


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(__file__)
    _, _, test_loader, classes = get_data_loaders(
        train_dir=os.path.join(base_dir, 'train_images'),
        test_dir=os.path.join(base_dir, 'test_images'),
        batch_size=64,
        num_workers=2,
    )

    checkpoint_path = os.path.join(base_dir, 'artifacts', 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint no encontrado en: {checkpoint_path}. Ejecuta primero el entrenamiento:")
        print(f"python {os.path.join(base_dir, 'train.py')}")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = Net().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

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
    print(f"Aciertos: {correct} de {total} ({acc*100:.2f}%)")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(classes))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    plt.savefig(os.path.join(artifacts_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # Visualize some predictions
    samples_to_show = 10
    shown = 0
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes = axes.ravel()

    # Create a small iterable of samples
    for images, labels in test_loader:
        with torch.no_grad():
            outputs = model(images.to(device))
            probs = F.softmax(outputs, dim=1).cpu()
            preds = probs.argmax(dim=1)
        for i in range(images.size(0)):
            if shown >= samples_to_show:
                break
            img = images[i].squeeze(0)  # 1x36x36 -> 36x36
            axes[shown].imshow(img.numpy() * 0.5 + 0.5, cmap='gray')
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

