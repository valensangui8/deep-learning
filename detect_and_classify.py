import os
import argparse
from typing import List, Tuple

import cv2
import torch
import torch.nn.functional as F

from utils import load_model_from_checkpoint, get_preprocess_transform


def detect_faces(image_bgr, scale_factor: float = 1.1, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def classify_crops(image_bgr, boxes: List[Tuple[int, int, int, int]], model, device: torch.device,
                   model_name: str, threshold: float = 0.5) -> List[Tuple[Tuple[int, int, int, int], int, float]]:
    transform = get_preprocess_transform(model_name, use_augmentation=False)
    results = []
    for (x, y, w, h) in boxes:
        crop = image_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(crop_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
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
    parser = argparse.ArgumentParser(description="Detect faces and classify them with the selected model")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("--model", type=str, default="baseline", choices=sorted(MODEL_REGISTRY.keys()),
                        help="Name of the model to use")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the model checkpoint (defaults to artifacts/<model>/best_model.pt)")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="scaleFactor parameter for the Haar detector")
    parser.add_argument("--min-neighbors", type=int, default=5, help="minNeighbors parameter for the Haar detector")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for 'face'")
    parser.add_argument("--save", type=str, default=None, help="Path to save the annotated image")
    parser.add_argument("--show", action="store_true", help="Show the annotated image in a window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(
        model_name=args.model,
        device=device,
        checkpoint_path=args.checkpoint,
        base_dir=os.path.dirname(__file__)
    )

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not open image: {args.image}")

    boxes = detect_faces(image_bgr, scale_factor=args.scale_factor, min_neighbors=args.min_neighbors)
    results = classify_crops(image_bgr, boxes, model, device, args.model, threshold=args.threshold)

    annotated = draw_annotations(image_bgr.copy(), results)

    if args.save is None:
        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        args.save = os.path.join(artifacts_dir, f"detections_{base}.png")

    cv2.imwrite(args.save, annotated)
    print(f"Results: {len(results)} detections. Image saved to: {args.save}")
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

