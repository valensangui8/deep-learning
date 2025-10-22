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
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, Tuple[str, str]]:
    """
    Create train/valid/test dataloaders for face detection dataset.

    Images are converted to grayscale and resized to 36x36 to match the CNN in net.py.
    Returns (train_loader, valid_loader, test_loader, classes).
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((36, 36)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

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

