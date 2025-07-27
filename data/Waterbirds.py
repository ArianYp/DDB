import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import warnings
warnings.filterwarnings("ignore")

class WaterbirdDataset(Dataset):
    def __init__(self, split,path, transform):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
            'retrain': 0
        }
        self.env_dict = {
            (0, 0): torch.Tensor(np.array([1,0,0,0])),#Landbird with land background
            (0, 1): torch.Tensor(np.array([0,1,0,0])),#Landbird with Water background
            (1, 0): torch.Tensor(np.array([0,0,1,0])),#Waterdbird with Land background
            (1, 1): torch.Tensor(np.array([0,0,0,1])) #Waterbird with water background
        }
        self.split = split
        self.dataset_dir= path
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[self.split]]


        y_array = torch.Tensor(np.array(self.metadata_df['y'].values)).type(torch.LongTensor)
        self.y_array = self.metadata_df['y'].values

        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = transform

        self.y_one_hot = nn.functional.one_hot(y_array, num_classes=2).type(torch.FloatTensor)
       
    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        label = self.y_one_hot[idx]

        return img, label, self.env_dict[(y, place)]


def get_waterbird_dataloaders(path, batch_size):
    kwargs = {'pin_memory': True, 'num_workers': 4, 'drop_last': False}

    traindataset = WaterbirdDataset( split='train',path= path, transform = get_transform_waterbirds(train=True))
    valdataset = WaterbirdDataset( split='val',path= path, transform = get_transform_waterbirds(train=False))
    testdataset = WaterbirdDataset( split='test',path= path, transform = get_transform_waterbirds(train=False))
    train_loadeer = DataLoader(dataset=traindataset,  batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=valdataset ,batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(dataset=testdataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loadeer, val_loader, test_loader

def get_waterbird_dataset(split,path, transform):
    dataset = WaterbirdDataset(split=split,path=path, transform = transform)
    return dataset

def get_transform_waterbirds(train):
    scale = 256.0/224.0
    target_resolution = [224, 224]
    assert target_resolution is not None

    if (not train):
      transform = transforms.Compose([
                transforms.Resize(
                    (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
            ])

    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                            0.229, 0.224, 0.225]),
        ])

    return transform
