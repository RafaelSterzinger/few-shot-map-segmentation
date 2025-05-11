r""" Dataloader builder for few-shot semantic segmentation dataset  """
import os
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import CutMix, RandomChoice, MixUp, Identity

from data.map_icdar import DatasetICDAR
from data.map_siegfried import DatasetSiegfried
from torch.utils.data import default_collate

def build_dataloader(args, transformer):
        dataloaders = [] 

        for split in ['train', 'val', 'test']:
                if args.class_name == 'icdar':
                    path = os.path.join('/data/databases', f'maps/maps_icdar/1-detbblocks')
                    dataset = DatasetICDAR(path, transformer, split, args.nshots, is_unet=True if args.base_model == 'unet' else False)
                else:
                    if args.nshots == 10 and split != 'test' and args.seed == 42:
                        path = os.path.join('/data/databases', f'maps/maps_siegfried/dataset_{args.class_name}/fewshot10')
                        print("Using predefined dataset with 10 shots from original paper")
                    elif split == 'val':
                        path = os.path.join('/data/databases', f'maps/maps_siegfried/dataset_{args.class_name}/fewshot10')
                    else:
                        path = os.path.join('/data/databases', f'maps/maps_siegfried/dataset_{args.class_name}')

                    dataset = DatasetSiegfried(path, transformer, split, args.nshots, is_unet=True if args.base_model == 'unet' else False)

                is_train = split == 'train'
                dataloaders.append(DataLoader(dataset, batch_size=args.batch_size if is_train else args.batch_size, shuffle=is_train, num_workers=8, drop_last=False))

        return dataloaders