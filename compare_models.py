import os
import argparse
from typing import Dict

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models import MODEL_REGISTRY


def get_preprocess_transform(model_name: str):
    pretrained_models = ['resnet18', 'mobilenetv2', 'efficientnet']
    if model_name in pretrained_models:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((36, 36)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])


def load_model(model_name: str, device: torch.device, checkpoint: str = None):
    if checkpoint is None:
        checkpoint = os.path.join(os.path.dirname(__file__), 'artifacts', model_name, 'best_model.pt')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint for '{model_name}' not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model = MODEL_REGISTRY[model_name]().to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Compare all models on an image')
    parser.add_argument('image', type=str, help='Path to the image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for face/noface decision')
    parser.add_argument('--save', action='store_true', help='Save per-model annotations in artifacts/')
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not open image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results: Dict[str, float] = {}
    for name in sorted(MODEL_REGISTRY.keys()):
        try:
            model = load_model(name, device)
        except FileNotFoundError:
            results[name] = float('nan')
            continue
        transform = get_preprocess_transform(name)
        tensor = transform(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            face_prob = float(probs[1].item())
            results[name] = face_prob

    print("Results (probability of 'face'):")
    for name, prob in results.items():
        if prob != prob:
            print(f"- {name}: checkpoint not found")
        else:
            label = 'face' if prob >= args.threshold else 'noface'
            print(f"- {name}: {prob:.3f} -> {label}")

    if args.save:
        artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_txt = os.path.join(artifacts_dir, f'compare_{base}.txt')
        with open(out_txt, 'w') as f:
            f.write("Results (face probability)\n")
            for name, prob in results.items():
                if prob != prob:
                    f.write(f"{name}: checkpoint not found\n")
                else:
                    label = 'face' if prob >= args.threshold else 'noface'
                    f.write(f"{name}: {prob:.4f} -> {label}\n")
        print(f"Summary saved to: {out_txt}")


if __name__ == '__main__':
    main()

