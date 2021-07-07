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
class NeuralNetwork(nn.module):
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
print(mlp)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=0.01)
