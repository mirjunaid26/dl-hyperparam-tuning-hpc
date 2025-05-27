import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# Flush function for safe logging
def log(msg):
    print(msg)
    sys.stdout.flush()

# Hyperparameters from environment variables
lr = float(os.environ.get("LR", 0.001))
batch_size = int(os.environ.get("BS", 32))
epochs = int(os.environ.get("EPOCHS", 5))

# Log job start
log(f"Starting training: LR={lr}, BS={batch_size}, Epochs={epochs}")

# Data preprocessing
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="/lustre/scratch/cbm107c-ai_llm/dl-hyperparam-tuning-hpc/mnist_data", train=True, download=False, transform=transform
)
test_dataset = datasets.MNIST(
    root="/lustre/scratch/cbm107c-ai_llm/dl-hyperparam-tuning-hpc/mnist_data", train=False, download=False, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    log(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

log(f"Test Accuracy: {100 * correct / total:.2f}%")
log("Training complete.")
