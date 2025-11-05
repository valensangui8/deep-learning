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
    parser.add_argument("--num-workers", type=int, default=2, help="Workers para DataLoader (CNNs pequeñas)")
    parser.add_argument("--pretrained-num-workers", type=int, default=0,
                        help="Workers para modelos preentrenados (0 para evitar problemas de shm)")
    parser.add_argument("--skip", type=str, default="",
                        help="Lista separada por comas de modelos a omitir")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Detener el entrenamiento si un modelo falla (por defecto continúa)")
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

    print(f"\n{'='*60}")
    print(f"Entrenando {len(models_to_train)} modelo(s): {', '.join(models_to_train)}")
    print(f"{'='*60}\n")

    failures = []
    successes = []
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n[{i}/{len(models_to_train)}] Entrenando modelo: {model_name}")
        batch_size = args.pretrained_batch_size if model_name in PRETRAINED_MODELS else args.batch_size
        num_workers = args.pretrained_num_workers if model_name in PRETRAINED_MODELS else args.num_workers
        
        cmd = [
            sys.executable, train_script,
            "--model", model_name,
            "--epochs", str(args.epochs),
            "--batch-size", str(batch_size),
            "--lr", str(args.lr),
            "--num-workers", str(num_workers),
        ]
        code = run(cmd)
        if code != 0:
            failures.append(model_name)
            if args.stop_on_error:
                print(f"\n❌ Error entrenando {model_name}. Abortando (--stop-on-error activado).")
                sys.exit(1)
            print(f"⚠️  {model_name} falló, pero continuando con los demás modelos...")
        else:
            successes.append(model_name)
            print(f"✅ {model_name} completado exitosamente")

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"✅ Exitosos: {len(successes)} - {', '.join(successes) if successes else 'ninguno'}")
    if failures:
        print(f"❌ Fallidos: {len(failures)} - {', '.join(failures)}")
        print(f"\n⚠️  Algunos modelos fallaron. Revisa los logs arriba para más detalles.")
        print(f"💡 Tip: Los modelos preentrenados pueden necesitar num_workers=0 en entornos con shm limitado.")
        sys.exit(1)
    else:
        print(f"\n🎉 Todos los modelos entrenaron correctamente!")


if __name__ == "__main__":
    main()


