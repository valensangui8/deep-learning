import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Tuple


def get_data_loaders(
    train_dir: str = './train_images',
    test_dir: str = './test_images',
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = 2,
    shuffle_dataset: bool = True,
    random_seed: int = 42,
    model_name: str = 'baseline',
    use_augmentation: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, Tuple[str, str]]:
    """
    Create train/valid/test dataloaders for face detection dataset.

    Args:
        model_name: Name of the model to determine input size and augmentation strategy
        use_augmentation: Whether to use data augmentation for training
    Returns (train_loader, valid_loader, test_loader, classes).
    """
    # Determine input size based on model
    pretrained_models = ['resnet18', 'mobilenetv2', 'efficientnet']
    if model_name in pretrained_models:
        input_size = 224
        # Normalization for ImageNet pretrained models (RGB 3 channels)
        # Note: We convert grayscale to RGB by duplicating channels, so we use RGB normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        input_size = 36
        # Normalization for grayscale (1 channel)
        normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    
    # Data augmentation for training
    if use_augmentation and model_name in pretrained_models:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB for pretrained models
            transforms.Resize((input_size + 32, input_size + 32)),  # Slightly larger for random crop
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif use_augmentation:
        # Light augmentation for small CNNs
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # No augmentation
        if model_name in pretrained_models:
            train_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                normalize,
            ])
    
    # Validation and test transforms (no augmentation)
    if model_name in pretrained_models:
        val_test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        val_test_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=val_test_transform)

    num_train = len(train_data)
    indices_train = list(range(num_train))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

    train_sampler = SubsetRandomSampler(train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    classes = ('noface', 'face')
    return train_loader, valid_loader, test_loader, classes

