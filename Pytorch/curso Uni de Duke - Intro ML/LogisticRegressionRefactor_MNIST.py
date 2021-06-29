import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

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
class MNIST_Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, x):
        return self.lin(x)

## Training
# Instance the model
model = MNIST_Logistic_Regression()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 1

for epoch in range(epochs):
    print('EPOCH: {} {}'.format(epoch, '-'*10))

    # Iterate through train set minibatchs
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images.view(-1, dim_image * dim_image) # flatten
        y = model(x)

        # Total loss
        loss = criterion(y, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

print("TRAINING COMPLETED")

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, dim_image * dim_image)
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))
