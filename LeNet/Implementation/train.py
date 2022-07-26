import torch
import torch.nn as nn
import numpy as np
from model import LeNet
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

lr = 0.1
batch_size = 64
num_epoch = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

training_data = datasets.MNIST(
    root="../../data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root="../../data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, num_workers=8)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

num_batch_train = np.ceil(len(training_data) / batch_size)

def train_model():
    for epoch in range(num_epoch):
        model.train()

        for batch, data in enumerate(train_dataloader, 1):
            X, y = data
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch + 1, num_epoch, batch, num_batch_train, loss.item()))

def test_model():
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        loss_arr = []

        for batch, (X, y) in enumerate(test_dataloader, 1):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            y_hat = net(X)
            _, y_hat = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (y_hat == y).sum().float()

        print("Accuracy of Test Data: {}%".format(100 * correct / total))

if __name__ == "__main__":
    train_model()
    test_model()