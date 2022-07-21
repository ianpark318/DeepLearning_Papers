## 필요한 패키지 등록
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, seed=10):
        self.path = path
        self.transform = transform
        self.seed = seed

        lst_data = os.listdir(self.path)

        lst_input = [f for f in lst_data if f.startswith('input')]
        lst_label = [f for f in lst_data if f.startswith('label')]

        lst_label.sort()
        lst_input.sort()

        self.lst_input = lst_input
        self.lst_label = lst_label

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, idx):
        label = np.load(os.path.join(self.path, self.lst_label[idx]))
        input = np.load(os.path.join(self.path, self.lst_input[idx]))

        '''
        label = label / 255.0
        input = input / 255.0


        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
        '''
        data = {'input': input, 'label': label}

        if self.transform:
            torch.manual_seed(self.seed)
            data['input'] = self.transform(data['input'])

        if self.transform:
            torch.manual_seed(self.seed)
            data['label'] = self.transform(data['label'])

        return data

class CarvanaDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).resize([960, 512]), dtype=np.float32)

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
            mask = np.array(Image.open(mask_path).resize([960, 512]), dtype=np.float32)
            #image = image.transpose(2, 0, 1)
            mask = mask[:, :, np.newaxis]
            mask[mask == 255.0] == 1.0

            data = {'input': image, 'label': mask}

            if self.transform:
                data['label'] = self.transform(data['label'])
                data['input'] = self.transform(data['input'])

        else:
            data = {'input': image}

            if self.transform:
                data['input'] = self.transform(data['input'])

        return data