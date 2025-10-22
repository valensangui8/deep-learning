import os
import time
import json
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from load_data import get_data_loaders
from net import Net


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(__file__)
    train_loader, valid_loader, test_loader, classes = get_data_loaders(
        train_dir=os.path.join(base_dir, 'train_images'),
        test_dir=os.path.join(base_dir, 'test_images'),
        batch_size=64,
        valid_size=0.2,
        num_workers=2,
    )

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_valid_acc = 0.0
    history: Dict[str, list] = {
        'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []
    }

    artifacts_dir = os.path.join(base_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, 'best_model.pt')
    history_path = os.path.join(artifacts_dir, 'history.json')

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save({'model_state_dict': model.state_dict(), 'classes': classes}, model_path)

        print(f"Epoch {epoch:02d}/{num_epochs} - "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
              f"valid_loss: {valid_loss:.4f} valid_acc: {valid_acc:.4f} "
              f"time: {time.time() - start:.1f}s")

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Final test eval on best checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test - loss: {test_loss:.4f} acc: {test_acc:.4f}")


if __name__ == '__main__':
    main()


