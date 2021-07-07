import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm, trange

# Load the data
dataset_url = '../../../../../ML_DL/datasets/mnist' # edit

if os.path.isdir(dataset_url):
    download = False
else:
    download = True
    dataset_url = './datasets'

mnist_train = datasets.MNIST(
    root=dataset_url,
    train=True,
    transform=transforms.ToTensor(),
    download=download
)
mnist_test = datasets.MNIST(
    root=dataset_url,
    train=False,
    transform=transforms.ToTensor(),
    download=download
)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

image, _ = mnist_train[0]
dim_image = image.shape[1]
num_classes = 10

# Define model

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # conv layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # conv layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # fc layer 1
        x = x.view(-1, 7*7*64)
        x = self.fc1(x)
        x = F.relu(x)

        # fc layer 2
        x = self.fc2(x)

        return x

model = MNIST_CNN()
print(model)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 3

# Iterative process
for epoch in trange(epochs):
    for images, labels in tqdm(train_loader):
        # zero output the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images
        y = model(x)
        loss = criterion(y, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

# Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        # forward pass
        x = images
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct / total))

# Test accuracy: 0.9918000102043152
