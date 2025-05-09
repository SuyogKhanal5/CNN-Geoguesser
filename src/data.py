from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch
import numpy as np


def calculate_mean_std(dataset):
    """Calculate mean and std of the dataset for normalization."""
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size
    
    mean /= total_images
    std /= total_images
    
    return mean, std


def load_from_folder(fname):
    # Define basic transformation for computing statistics
    basic_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create temporary dataset for calculating statistics
    temp_dataset = datasets.ImageFolder(root=fname, transform=basic_transform)
    
    # Calculate mean and std
    mean, std = calculate_mean_std(temp_dataset)
    print(f"Dataset mean: {mean}, std: {std}")
    
    # Define the final transformation with calculated normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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
