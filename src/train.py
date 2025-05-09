import torch
import torch.nn as nn
import torch.optim as optim
import data as data
from model import CNNModel
import wandb

# Hyperparameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = 'data'
MODEL_SAVE_PATH = 'cnn_geoguesser_model.pth'

wandb.init(project="cnn-geoguesser", config={
    "learning_rate": LEARNING_RATE,
    "epochs": NUM_EPOCHS,
    "batch_size": 32, # Assuming batch_size is 32 from DataLoader
    "architecture": "CNNModel",
    "dataset": "CustomGeoImages"
})
config = wandb.config

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
print("Loading data...")
train_loader, val_loader, class_to_idx = data.load_from_folder(DATA_DIR)
num_classes = len(class_to_idx)
print(f"Found {num_classes} classes: {class_to_idx}")

# Initialize model, loss, and optimizer
print("Initializing model...")
model = CNNModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Watch the model with wandb
wandb.watch(model, log="all")

print("Starting training...")
best_val_accuracy = 0.0

for epoch in range(config.epochs):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Log batch loss to wandb
        wandb.log({"batch_loss": loss.item()})

        if (i + 1) % 10 == 0: # Print every 10 batches
                print(f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = 100 * train_correct / train_total
    # Log epoch metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "train_accuracy": epoch_train_acc
    })
    print(f'Epoch {epoch+1} Training Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%')


    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = 100 * val_correct / val_total
    # Log validation metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "val_loss": epoch_val_loss,
        "val_accuracy": epoch_val_acc
    })
    print(f'Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.2f}%')

    # Save the best model
    if epoch_val_acc > best_val_accuracy:
        best_val_accuracy = epoch_val_acc
        torch.save(model, MODEL_SAVE_PATH)
        print(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")

print('Finished Training')
# Finish the wandb run
wandb.finish()
print(f"Model saved to {MODEL_SAVE_PATH}")