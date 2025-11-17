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
    parser = argparse.ArgumentParser(description="Train all models or a specific subset")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated list of models to train or 'all' for every model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for every model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for small CNNs")
    parser.add_argument("--pretrained-batch-size", type=int, default=32,
                        help="Batch size for pretrained models")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers for small CNNs")
    parser.add_argument("--pretrained-num-workers", type=int, default=0,
                        help="Workers for pretrained models (0 to avoid shm issues)")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated list of models to skip")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop training if a model fails (continues by default)")
    args = parser.parse_args()

    available = list(sorted(MODEL_REGISTRY.keys()))

    if args.models.lower() == "all":
        models_to_train = available
    else:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in available]
        if unknown:
            print(f"Unknown models: {unknown}")
            print(f"Available: {available}")
            sys.exit(1)
        models_to_train = requested

    if args.skip:
        skip = {m.strip() for m in args.skip.split(",") if m.strip()}
        models_to_train = [m for m in models_to_train if m not in skip]

    project_root = os.path.dirname(__file__)
    train_script = os.path.join(project_root, "train.py")

    print(f"\n{'='*60}")
    print(f"Training {len(models_to_train)} model(s): {', '.join(models_to_train)}")
    print(f"{'='*60}\n")

    failures = []
    successes = []
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n[{i}/{len(models_to_train)}] Training model: {model_name}")
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
                print(f"\n❌ Error training {model_name}. Aborting (--stop-on-error enabled).")
                sys.exit(1)
            print(f"⚠️  {model_name} failed, continuing with the remaining models...")
        else:
            successes.append(model_name)
            print(f"✅ {model_name} completed successfully")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successes)} - {', '.join(successes) if successes else 'none'}")
    if failures:
        print(f"❌ Failed: {len(failures)} - {', '.join(failures)}")
        print(f"\n⚠️  Some models failed. Check the logs above for details.")
        print(f"💡 Tip: Pretrained models may need num_workers=0 when shm is limited.")
        sys.exit(1)
    else:
        print(f"\n🎉 All models trained successfully!")


if __name__ == "__main__":
    main()
