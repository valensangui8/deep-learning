import os
import argparse
from typing import Dict

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models import MODEL_REGISTRY


def get_preprocess_transform(model_name: str):
    """Get preprocessing transform based on model type"""
    pretrained_models = ['resnet18', 'mobilenetv2', 'efficientnet']
    if model_name in pretrained_models:
        # Pretrained models: 224x224 RGB with ImageNet normalization
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Small CNNs: 36x36 grayscale
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
        raise FileNotFoundError(f"Checkpoint para '{model_name}' no encontrado: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model = MODEL_REGISTRY[model_name]().to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Comparar todos los modelos sobre una imagen')
    parser.add_argument('image', type=str, help='Ruta a la imagen')
    parser.add_argument('--threshold', type=float, default=0.5, help='Umbral para decidir face/noface')
    parser.add_argument('--save', action='store_true', help='Guardar anotaciones por modelo en artifacts/')
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results: Dict[str, float] = {}
    for name in sorted(MODEL_REGISTRY.keys()):
        try:
            model = load_model(name, device)
        except FileNotFoundError:
            results[name] = float('nan')
            continue
        # Get correct transform for this model
        transform = get_preprocess_transform(name)
        tensor = transform(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            face_prob = float(probs[1].item())
            results[name] = face_prob

    # Mostrar resumen
    print("Resultados (probabilidad de 'face'):")
    for name, prob in results.items():
        if prob != prob:  # NaN check
            print(f"- {name}: checkpoint no encontrado")
        else:
            label = 'face' if prob >= args.threshold else 'noface'
            print(f"- {name}: {prob:.3f} -> {label}")

    # Guardado simple por conveniencia (opcional): escribe resultados en artifacts
    if args.save:
        artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_txt = os.path.join(artifacts_dir, f'compare_{base}.txt')
        with open(out_txt, 'w') as f:
            f.write("Resultados (prob face)\n")
            for name, prob in results.items():
                if prob != prob:
                    f.write(f"{name}: checkpoint no encontrado\n")
                else:
                    label = 'face' if prob >= args.threshold else 'noface'
                    f.write(f"{name}: {prob:.4f} -> {label}\n")
        print(f"Resumen guardado en: {out_txt}")


if __name__ == '__main__':
    main()


