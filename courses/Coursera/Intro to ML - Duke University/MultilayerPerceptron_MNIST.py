import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import  datasets, transforms
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

# Define model - 2-layer MLP (Input - 784, FCL 500 units [ReLu], FCL 10 units [Softmax])
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=dim_image*dim_image, out_features=500)
        self.fc1 = nn.Linear(in_features=500, out_features=num_classes)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        return self.fc1(x)

# Instance model
mlp = NeuralNetwork()

# Sumary of model
print(mlp)

model_parameters = filter(lambda p: p.requires_grad, mlp.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1)

# Training
epochs = 3

for epoch in range(epochs):
    print('EPOCH: {}/{} {}'.format(epoch + 1, epochs, '-'*10))

    # Iterate through train set minibatchs
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images.view(-1, dim_image * dim_image) # flatten
        y = mlp(x)

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
        y = mlp(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))
