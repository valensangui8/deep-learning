import os
import argparse
from typing import List, Tuple

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from net import Net


def load_model(checkpoint_path: str, device: torch.device) -> Net:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint no encontrado en: {checkpoint_path}. Ejecuta primero el entrenamiento."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Net().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_preprocess_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((36, 36)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])


def detect_faces(image_bgr, scale_factor: float = 1.1, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def classify_crops(image_bgr, boxes: List[Tuple[int, int, int, int]], model: Net, device: torch.device,
                   threshold: float = 0.5) -> List[Tuple[Tuple[int, int, int, int], int, float]]:
    transform = get_preprocess_transform()
    results = []
    for (x, y, w, h) in boxes:
        crop = image_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(crop_rgb).unsqueeze(0).to(device)  # (1,1,36,36)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            # clases: ('noface', 'face') -> índice 1 es 'face'
            face_prob = float(probs[1].item())
            pred = 1 if face_prob >= threshold else 0
        results.append(((x, y, w, h), pred, face_prob))
    return results


def draw_annotations(image_bgr, results: List[Tuple[Tuple[int, int, int, int], int, float]]):
    for (x, y, w, h), pred, prob in results:
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)
        label = f"{'face' if pred == 1 else 'noface'} {prob:.2f}"
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_bgr, (x, y - th - baseline), (x + tw, y), color, -1)
        cv2.putText(image_bgr, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return image_bgr


def main():
    parser = argparse.ArgumentParser(description="Detectar caras y clasificarlas usando el modelo existente")
    parser.add_argument("image", type=str, help="Ruta a la imagen de entrada")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(os.path.dirname(__file__), "artifacts", "best_model.pt"),
                        help="Ruta al checkpoint del modelo")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Parámetro scaleFactor del detector Haar")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Parámetro minNeighbors del detector Haar")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de probabilidad para 'face'")
    parser.add_argument("--save", type=str, default=None, help="Ruta para guardar la imagen anotada")
    parser.add_argument("--show", action="store_true", help="Mostrar la imagen anotada en una ventana")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {args.image}")

    boxes = detect_faces(image_bgr, scale_factor=args.scale_factor, min_neighbors=args.min_neighbors)
    results = classify_crops(image_bgr, boxes, model, device, threshold=args.threshold)

    annotated = draw_annotations(image_bgr.copy(), results)

    # Guardar por defecto en artifacts si no se especifica
    if args.save is None:
        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        args.save = os.path.join(artifacts_dir, f"detections_{base}.png")

    cv2.imwrite(args.save, annotated)
    print(f"Resultados: {len(results)} detecciones. Imagen guardada en: {args.save}")
    for (x, y, w, h), pred, prob in results:
        print(f"bbox=({x},{y},{w},{h}) label={'face' if pred==1 else 'noface'} prob={prob:.3f}")

    if args.show:
        try:
            cv2.imshow("detections", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()


