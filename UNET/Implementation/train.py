import os
import numpy as np
import torch
import torch.nn as nn
from model import UNET
from my_data_load import Dataset, CarvanaDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

lr = 1e-3
batch_size = 2
num_epoch = 5



#data_dir = './datasets'
train_dir = './train'
mask_dir = './train_masks'
test_dir = './test'
ckpt_dir = './checkpoint'
log_dir = './log'
result_dir = './result'

my_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

#dataset_train = Dataset(path=os.path.join(data_dir, 'train'), transform=my_transform)
dataset_train = CarvanaDataset(image_dir=train_dir, mask_dir=mask_dir, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=0.0, std=255.0)]))
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)

'''
dataset_val = Dataset(path=os.path.join(data_dir, 'val'), transform=my_transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
'''


num_batch_train = np.ceil(len(dataset_train) / batch_size)

#num_batch_val = np.ceil(len(dataset_val) / batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = UNET(in_channels=3).to(device)

fn_loss = nn.BCEWithLogitsLoss().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)
st_epoch = 0

def train_model():
    for epoch in range(st_epoch, num_epoch):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            data['label'] = data['label'] * 255.0
            label = data['label'].to(device)
            input = data['input'].to(device)

            out = net(input)
            # backward pass
            optimizer.zero_grad()

            loss = fn_loss(out, label)
            loss.backward()

            optimizer.step()

            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch+1, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
        '''
        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                data['label'] = data['label'] * 0.5 + 0.5
                label = data['label'].to(device)
                input = data['input'].to(device)

                out = net(input)

                loss = fn_loss(out, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch + 1, num_epoch, batch, num_batch_val, np.mean(loss_arr)))
            '''
def test_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.0, std=255.0)
    ])

    #dataset_test = Dataset(path=os.path.join(data_dir, 'test'), transform=transform)
    dataset_test = CarvanaDataset(image_dir=test_dir, transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    num_batch_test = np.ceil(len(dataset_test) / batch_size)

    print(type(dataset_test))

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기

            print("TEST: BATCH %04d / %04d" %
                  (batch, num_batch_test))

            input = fn_tonumpy(input)
            output = fn_tonumpy(fn_class(output))

            for j in range(input.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

    print("AVERAGE TEST: BATCH %04d / %04d" %
          (batch, num_batch_test))


if __name__ == "__main__":
    train_model()
    test_model()