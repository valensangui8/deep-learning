"""
Common utilities for model loading, preprocessing, and evaluation
Centralizes logic to ensure consistency across all scripts
"""
import os
from typing import Tuple, Optional
import torch
import torchvision.transforms as transforms

from models import MODEL_REGISTRY

# Constants
PRETRAINED_MODELS = {'resnet18', 'mobilenetv2', 'efficientnet'}


def get_preprocess_transform(model_name: str, use_augmentation: bool = False) -> transforms.Compose:
    """
    Get preprocessing transform for a model.
    This matches the test/validation transform from load_data.py to ensure consistency.
    
    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY)
        use_augmentation: Whether to apply augmentation (only used for training)
    
    Returns:
        transforms.Compose: Preprocessing pipeline
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(MODEL_REGISTRY.keys())}")
    
    is_pretrained = model_name in PRETRAINED_MODELS
    
    if is_pretrained:
        input_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        input_size = 36
        normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    
    # For detection/inference: no augmentation (matches test/val transforms from load_data.py)
    if not use_augmentation:
        if is_pretrained:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        # For training: with augmentation (matches train transforms from load_data.py)
        if is_pretrained:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                normalize,
            ])


def load_model_from_checkpoint(
    model_name: str,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    base_dir: Optional[str] = None
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY)
        device: torch.device to load model on
        checkpoint_path: Optional path to checkpoint. If None, uses default: artifacts/<model>/best_model.pt
        base_dir: Base directory for default checkpoint path (defaults to script directory)
    
    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(MODEL_REGISTRY.keys())}")
    
    if checkpoint_path is None:
        if base_dir is None:
            base_dir = os.path.dirname(__file__)
        checkpoint_path = os.path.join(base_dir, 'artifacts', model_name, 'best_model.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found for '{model_name}': {checkpoint_path}\n"
            f"Train the model first: python train.py --model {model_name}"
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass().to(device)
    
    # Handle both old and new checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback for old format (direct state dict)
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def get_checkpoint_path(model_name: str, base_dir: Optional[str] = None) -> str:
    """
    Get the default checkpoint path for a model.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory (defaults to script directory)
    
    Returns:
        str: Path to checkpoint file
    """
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, 'artifacts', model_name, 'best_model.pt')


def is_pretrained_model(model_name: str) -> bool:
    """Check if a model is a pretrained model."""
    return model_name in PRETRAINED_MODELS


def evaluate_on_loader(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                      device: torch.device) -> Tuple[list, list]:
    """
    Evaluate a model on a dataloader and return predictions and labels.
    
    Args:
        model: PyTorch model in eval mode
        dataloader: DataLoader with test/validation data
        device: torch.device
    
    Returns:
        Tuple of (all_labels, all_preds) as lists
    """
    import torch.nn.functional as F
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())
    return all_labels, all_preds

