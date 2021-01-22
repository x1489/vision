from __future__ import print_function
from __future__ import division

import os
import PIL
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50

from vit_pytorch.distill import DistillableViT, DistillWrapper

TRAIN_DIR = '../data/GLDv2/train'
TRAIN_LABEL_MAP_PATH = os.path.join(TRAIN_DIR, 'train_clean_reshaped.csv')

device = torch.device('cuda')

class GLD(Dataset):

    def __init__(self, image_dir, label_map_path, image_ext='.jpg'):

        self.df = pd.read_csv(label_map_path)
        self.image_dir = image_dir
        self.images = self.df.id.values
        self.labels = self.df.landmark_id.values
        self.suffix = image_ext

    def __getitem__(self, idx):
        img_id = self.images[idx]
        image = self.get_image_tensor(img_id, self.image_dir, self.suffix)
        image = T.functional.resize(image, [512, 512])
        label = torch.tensor(self.labels[idx])
        feature_dict = {'idx': torch.tensor(idx).long(),
                        'input': image, 'label': label}
        return feature_dict

    def __len__(self):
        return len(self.images)

    def get_image_tensor(self, id, img_dir, ext):
        try:
            img_path = img_dir + f'/{id[0]}/{id[1]}/{id[2]}/{id}{ext}'
            img = PIL.Image.open(img_path).convert('RGB')
            img = T.functional.to_tensor(img)
        except:
            print('FAILED READING IMG', f'{id}{ext}')
            img = torch.zeros(3, 512, 512)
        return img/255

    def normalize(self, img):
        mean = torch.tensor([123.675, 116.28, 103.53])
        std = torch.tensor([58.395, 57.120, 57.375])
        print(img.size(), mean.size())
        return img

trainset = GLD(TRAIN_DIR, TRAIN_LABEL_MAP_PATH)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

teacher  = resnet50(pretrained=True)

v = DistillableViT(
    image_size = 512,
    patch_size = 64,
    num_classes = 203092,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1)

distiller = DistillWrapper(
    student = v,
    teacher = teacher,
    temperature = 3,           # temperature of distillation
    alpha = 0.5)                # trade between main loss and distillation loss

v = nn.DataParallel(v)
v.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(v.parameters(), lr=1e-3)

try:
    for epoch in range(10):
        run_loss, epoch_loss = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            images, labels = data['input'].to(device), data['label'].to(device)
            preds = v(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            epoch_loss += loss.item()
            if i % 9 == 0:
                print(f'[Epoch {epoch + 1} | Step {i + 1} | Loss: {run_loss/(i + 1)}]')
                run_loss = 0
except KeyboardInterrupt:
    print(f'[Epoch {epoch + 1} | Epoch Loss: {epoch_loss/(i + 1)}]')
    torch.save(v, './saved_model.pt')

