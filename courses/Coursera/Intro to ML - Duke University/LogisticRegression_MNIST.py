import os

import numpy as np
import torch
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

## Training
# Initialize parameters
W = torch.randn(dim_image * dim_image, num_classes) / np.sqrt(784)
W.requires_grad_()
b = torch.zeros(num_classes, requires_grad=True)

# Optimizer
optimizer = torch.optim.SGD([W, b], lr=0.1)

epochs = 1

for epoch in range(epochs):
    print('EPOCH: {} {}'.format(epoch, '-'*10))

    # Iterate through train set minibatchs
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images.view(-1, dim_image * dim_image) # flatten
        y = torch.matmul(x, W) + b

        # Soft max (implicit, probabilities) and cross entropy (total loss)
        cross_entropy = F.cross_entropy(input=y, target=labels)

        # Backward pass
        cross_entropy.backward()
        optimizer.step()

print("TRAINING COMPLETED")

# Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, dim_image * dim_image)
        y = torch.matmul(x, W) + b

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))
