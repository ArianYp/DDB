import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random

class CelebADataset(Dataset):
    def __init__(self, phase, root_dir, transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        # (y, gender)
        self.env_dict = {
            (0, 0): torch.Tensor([1, 0, 0, 0]).type(torch.LongTensor),   # nonblond hair, female
            (0, 1): torch.Tensor([0, 1, 0, 0]).type(torch.LongTensor),   # nonblond hair, male
            (1, 0): torch.Tensor([0, 0, 1, 0]).type(torch.LongTensor),   # blond hair, female
            (1, 1): torch.Tensor([0, 0, 0, 1]).type(torch.LongTensor)    # blond hair, male
        }
        self.dataset_dir = root_dir
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celeba_split.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[phase]]



        if phase == 'train':
            label0 = self.metadata_df.index[self.metadata_df['Blond_Hair'] == 0].tolist()
            label1 = self.metadata_df.index[self.metadata_df['Blond_Hair'] == 1].tolist()
            l = int(min(len(label0), len(label1)))
            label0 = [label0[i]for i in random.sample(range(0, len(label0)), l)]
            label1 = [label1[i] for i in random.sample(range(0, len(label1)), l)]

            self.metadata_df = self.metadata_df.iloc[label0+label1]


        self.y_array = self.metadata_df['Blond_Hair'].values
        y_array = torch.Tensor(np.array(self.metadata_df['Blond_Hair'].values)).type(torch.LongTensor)
        self.y_one_hot = nn.functional.one_hot(y_array, num_classes=2).type(torch.FloatTensor)
        self.gender_array = self.metadata_df['Male'].values
        self.filename_array = self.metadata_df['image_id'].values
        self.transform = transform
        self.labels = torch.nn.functional.one_hot(torch.Tensor(self.y_array).type(torch.LongTensor)).type(torch.FloatTensor)

        
    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        gender = self.gender_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        env = self.env_dict[(y, gender)]

        return img, label, env


def get_celeba_dataset (phase, root_dir, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])):
    dataset = CelebADataset(phase=phase, root_dir=root_dir, transform=transform)
    return dataset

def get_loader (root_dir, phase, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))]), batch_size = 32):
    if phase == 'retrain':
        dataset = get_celeba_dataset('train', root_dir, transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4), dataset
    else:
        dataset = get_celeba_dataset(phase, root_dir, transform)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)



def get_transform_celeba(train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if not train:
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform

def get_celeba_loaders(path, batch_size):
    transforms_train = get_transform_celeba(True)
    transforms_test = get_transform_celeba(False)

    trainloader = get_loader(path, 'train', transform=transforms_train, batch_size=batch_size)
    valloader = get_loader(path, 'val', transform=transforms_test, batch_size=batch_size)
    testloader = get_loader(path, 'test', transform=transforms_test, batch_size=batch_size)

    return trainloader, valloader, testloader
