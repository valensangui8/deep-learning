import argparse
import os
import subprocess
import sys
from typing import List

from models import MODEL_REGISTRY


PRETRAINED_MODELS = {"resnet18", "mobilenetv2", "efficientnet"}


def run(cmd: List[str]) -> int:
    print("\n==>", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Entrenar todos o un subconjunto de modelos")
    parser.add_argument("--models", type=str, default="all",
                        help="Lista separada por comas de modelos a entrenar o 'all' para todos")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas para todos los modelos")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size para CNNs pequeñas")
    parser.add_argument("--pretrained-batch-size", type=int, default=32,
                        help="Batch size para modelos preentrenados")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers para DataLoader")
    parser.add_argument("--skip", type=str, default="",
                        help="Lista separada por comas de modelos a omitir")
    args = parser.parse_args()

    available = list(sorted(MODEL_REGISTRY.keys()))

    if args.models.lower() == "all":
        models_to_train = available
    else:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in available]
        if unknown:
            print(f"Modelos desconocidos: {unknown}")
            print(f"Disponibles: {available}")
            sys.exit(1)
        models_to_train = requested

    if args.skip:
        skip = {m.strip() for m in args.skip.split(",") if m.strip()}
        models_to_train = [m for m in models_to_train if m not in skip]

    project_root = os.path.dirname(__file__)
    train_script = os.path.join(project_root, "train.py")

    failures = []
    for model_name in models_to_train:
        batch_size = args.pretrained_batch_size if model_name in PRETRAINED_MODELS else args.batch_size
        cmd = [
            sys.executable, train_script,
            "--model", model_name,
            "--epochs", str(args.epochs),
            "--batch-size", str(batch_size),
            "--lr", str(args.lr),
            "--num-workers", str(args.num_workers),
        ]
        code = run(cmd)
        if code != 0:
            failures.append(model_name)

    if failures:
        print("\nAlgunos entrenamientos fallaron:")
        for m in failures:
            print(f"- {m}")
        sys.exit(1)
    else:
        print("\nTodos los modelos entrenaron correctamente.")


if __name__ == "__main__":
    main()


