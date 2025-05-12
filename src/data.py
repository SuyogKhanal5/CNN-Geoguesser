import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np


def load_from_folder(fname):
    # Define mean and std for normalization
    mean = torch.tensor([0.5187, 0.5390, 0.5109])
    std = torch.tensor([0.1955, 0.1946, 0.2522])
    print(f"Using pre-calculated Dataset mean: {mean}, std: {std}")

    # Define the final transformation with calculated normalization
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load the dataset with proper normalization
    dataset = datasets.ImageFolder(root=fname, transform=transform)
    class_to_idx = dataset.class_to_idx

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, class_to_idx

load_from_folder("streetview_dataset")