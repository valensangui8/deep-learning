import argparse
import os
from typing import Dict, List

import cv2
import torch

from models import MODEL_REGISTRY
from detect_and_classify import (
    load_model,
    detect_faces,
    classify_crops,
    draw_annotations,
)


def run_detection_for_model(
    image_path: str,
    model_name: str,
    device: torch.device,
    checkpoint: str,
    scale_factor: float,
    min_neighbors: int,
    threshold: float,
    output_dir: str,
) -> Dict[str, List]:
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(checkpoint, device, model_name)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    boxes = detect_faces(image_bgr, scale_factor=scale_factor, min_neighbors=min_neighbors)
    results = classify_crops(image_bgr, boxes, model, device, model_name, threshold=threshold)

    annotated = draw_annotations(image_bgr.copy(), results)
    output_path = os.path.join(output_dir, f"{model_name}.png")
    cv2.imwrite(output_path, annotated)

    summary = {
        "model": model_name,
        "num_detections": len(results),
        "detections": results,
        "output_path": output_path,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Ejecutar detección con todos los modelos entrenados")
    parser.add_argument("image", type=str, help="Ruta a la imagen de entrada")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Lista separada por comas de modelos a usar o 'all' para todos",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Parámetro scaleFactor del detector Haar",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Parámetro minNeighbors del detector Haar",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de probabilidad para clasificar como 'face'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio base para guardar las imágenes anotadas (por defecto artifacts/detections/<imagen>)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Omitir modelos sin checkpoint en lugar de detenerse con error",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.models.lower() == "all":
        model_names = sorted(MODEL_REGISTRY.keys())
    else:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in MODEL_REGISTRY]
        if unknown:
            raise ValueError(f"Modelos desconocidos: {unknown}")
        model_names = requested

    image_base = os.path.splitext(os.path.basename(args.image))[0]
    if args.output_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "artifacts", "detections", image_base)
    else:
        base_dir = args.output_dir

    os.makedirs(base_dir, exist_ok=True)

    summaries: List[Dict] = []
    failures: List[str] = []

    for model_name in model_names:
        checkpoint = os.path.join(
            os.path.dirname(__file__),
            "artifacts",
            model_name,
            "best_model.pt",
        )
        if not os.path.exists(checkpoint):
            msg = f"Checkpoint no encontrado para {model_name}: {checkpoint}"
            if args.skip_missing:
                print(f"⚠️  {msg}. Omitiendo...")
                failures.append(model_name)
                continue
            raise FileNotFoundError(msg)

        print(f"\n==> Ejecutando detección con modelo: {model_name}")
        try:
            summary = run_detection_for_model(
                image_path=args.image,
                model_name=model_name,
                device=device,
                checkpoint=checkpoint,
                scale_factor=args.scale_factor,
                min_neighbors=args.min_neighbors,
                threshold=args.threshold,
                output_dir=base_dir,
            )
            summaries.append(summary)
            print(
                f"   Detecciones: {summary['num_detections']} | salida: {summary['output_path']}"
            )
        except Exception as exc:
            failures.append(model_name)
            if args.skip_missing:
                print(f"⚠️  Error con {model_name}: {exc}. Omitiendo...")
                continue
            raise

    print("\nResumen:")
    for summary in summaries:
        print(
            f"- {summary['model']}: {summary['num_detections']} detecciones "
            f"(imagen -> {summary['output_path']})"
        )

    if failures:
        print("\nModelos con problemas u omitidos:")
        for model_name in failures:
            print(f"- {model_name}")
    else:
        print("\nTodos los modelos procesaron la imagen correctamente.")


if __name__ == "__main__":
    main()


