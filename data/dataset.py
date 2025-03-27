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

from data.text_icdar import DatasetTextICDAR

PATH_DICT = {'latin1' : 'Latin14396', 'latin2' : 'Latin2', 'syr' : 'Syr341'}

def build_dataloader(args, transformer, use_mixup=False, use_cutmix=False):
        dataloaders = [] 

        cutmix = CutMix(num_classes=1)
        mixup = MixUp(num_classes=1)

        def collate_fn(batch):
            B = batch.__len__()
            temp = default_collate(batch)
            # temp[0]==image, temp[1]==label
            temp = [temp['img'], temp['mask'].unsqueeze(1)]
            # stack image and labels on the channel dim, i.e. gt map
            results=[]
            stack = torch.cat(temp, dim=1)
            results.append(stack)
            if use_mixup:       
                results.append(mixup(
                stack, torch.zeros(stack.shape[0], dtype=int))[0])
            if use_cutmix:

                results.append(cutmix(
                stack, torch.zeros(stack.shape[0], dtype=int))[0])
            
            output = torch.stack([random.choice(results)[i] for i in range(B)], dim=0)

            # split image and labels again
            batched = {
                   'img':output[:, 0:-1],
                   'mask':(output[:, -1:].squeeze(1) > 0).float()
            }
            return batched

        for split in ['train', 'val', 'test']:
                if args.class_name == 'icdar':
                    path = os.path.join('/data/databases', f'maps/maps_icdar/1-detbblocks')
                    dataset = DatasetICDAR(path, transformer, split, args.nshots, is_unet=True if args.base_model == 'unet' else False)
                elif args.class_name in ['latin1', 'latin2', 'syr']:
                    if split == 'test':
                         split = 'val'
                    path = os.path.join('/data/databases', f'maps/text_icdar/{PATH_DICT[args.class_name]}')
                    dataset = DatasetTextICDAR(path, transformer, split, args.nshots, is_unet=True if args.base_model == 'unet' else False)
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