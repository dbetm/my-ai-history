import agnews

from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext


# SIMPLE WORD EMBEDDING MODEL
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
VOCAB_SIZE = len(agnews.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
NUM_OUTPUTS = agnews.num_classes
NUM_EPOCHS = 3

print('Number of training examples: ', len(agnews.train_loader))
print('Vocab size: ', VOCAB_SIZE)
print('Embedding dimensions: ', EMBEDDING_DIM)
print('Hidden dim: ', HIDDEN_DIM)
print('Num outputs/classes: ', NUM_OUTPUTS)


class SWEM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super(SWEM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size) # codebook

        self.fc1 = nn.Linear(in_features=embedding_size, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_outputs)

    def forward(self, x):
        embed = self.embedding(x)
        embed_mean = torch.mean(embed, dim=0)

        h = self.fc1(embed_mean)
        h = F.relu(h)
        h = self.fc2(h)

        return h

# Training
model = SWEM(
    vocab_size=VOCAB_SIZE,
    embedding_size=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_outputs=NUM_OUTPUTS
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in trange(NUM_EPOCHS):
    correct = 0
    num_examples = 0

    for X, Y in agnews.train_loader:
        # zero out gradients
        optimizer.zero_grad()

        # forward pass
        Y_pred = model(X)

        loss = criterion(Y_pred, Y)

        # backward pass
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(torch.sigmoid(Y_pred), axis=1)
        correct += torch.sum((predictions == Y).float())
        num_examples += len(Y)

    acc = correct / num_examples
    print("Epoch: {0} \t Train Loss: {1} \t Train Acc: {2}".format(epoch, loss, acc))

# Train Acc: 0.919783353805542
# Train Loss: 0.09890331327915192

# Testing
correct = 0
num_test = 0

with torch.no_grad():
    for X, Y in agnews.test_loader:
        # forward pass
        Y_pred = model(X)

        predictions = torch.argmax(torch.sigmoid(Y_pred), axis=1)
        correct += torch.sum((predictions == Y).float())

        num_test += len(Y)

print('\nTest accuracy: {}'.format(correct / num_test))

# Test accuracy: 0.9081578850746155
