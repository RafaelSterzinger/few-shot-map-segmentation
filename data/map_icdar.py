from math import ceil
import os
from torch.utils.data import Dataset
from patchify import patchify, unpatchify
import torch
import PIL.Image as Image
import numpy as np
import albumentations as A

from data.map_siegfried import DatasetSiegfried


class DatasetICDAR(Dataset):
    def __init__(self, datapath, transform, split, nshots = 1.0, is_unet=False):
        print('Loading data from %s' % datapath)
        self.split = split
        self.benchmark = 'maps_icdar'
        self.base_path = datapath
        self.transform = transform
        self.is_unet = is_unet
        self.images = self.load_images(self.split)


        assert nshots == 1.0

        scaling = 2
        patch_size = 224*scaling
        if split == 'train':
            self.augmentations = A.Compose([
                    A.CropNonEmptyMaskIfExists(patch_size, patch_size),
                    A.Resize(patch_size//scaling, patch_size//scaling),
                    A.D4()
                ])
        else:
            for i in range(len(self.images)):
                orig_size = self.images[i]['orig_size']
                transform = A.Compose([
                    A.PadIfNeeded(min_height=patch_size*(ceil(orig_size[0]/patch_size)), min_width=patch_size*(ceil(orig_size[1]/patch_size)), border_mode=0, position='top_left'),  
                ])
                new_size = None
                patch_shape = None
                for key,value in self.images[i].items(): 
                    if key not in ['input', 'gt']:
                        continue
                    value = transform(image=value)['image']
                    if new_size is None:
                        new_size = value.shape
                    self.images[i][key] = patchify(value, (patch_size,patch_size, 3) if key == 'input' else (patch_size,patch_size), step=patch_size)
                    shape = self.images[i][key].squeeze().shape
                    if patch_shape is None:
                        patch_shape = shape
                    self.images[i][key] = self.images[i][key].squeeze().reshape(-1, *shape[2:])
                self.images[i]['new_size'] = new_size
                self.images[i]['patch_shape'] = patch_shape
            self.start_indices = [0]  # Track where each list starts in global indexing
            self.length = 0
            for lst in self.images:
                self.start_indices.append(self.start_indices[-1] + len(lst['input']))
                self.length += len(lst['input'])

            self.augmentations = A.Compose([
                        A.Resize(patch_size//scaling, patch_size//scaling),
                    ])

    def global_to_local(self, global_index):
        """Finds the correct list and local index for a given global index."""
        for list_idx in range(len(self.images)):
            if global_index < self.start_indices[list_idx + 1]:  
                local_index = global_index - self.start_indices[list_idx]
                return list_idx, local_index
        raise IndexError("Global index out of range")

    def __len__(self):
        return 160 if self.split == 'train' else self.length

    def __getitem__(self, idx):
        image, mask, name = self.load_frame(idx)

        if self.is_unet:
            image = self.transform(image)
        else:
            image = self.transform(image)['pixel_values'][0]

        batch = {'img': image,
                 'mask': mask,
                 'name': name,
                 }

        return batch

    def load_images(self, split):
        def read_metadata(split):
            data = os.listdir(os.path.join(self.base_path, split))
            fold_n_metadata = [d.split('/')[-1].split('.')[0] for d in data]
            return set(map(lambda x: x[0:3], fold_n_metadata))

        img_metadata = read_metadata(split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        images = []
        for img in img_metadata:
            img_dict = {}
            base_path = os.path.join(self.base_path, split)
            img_dict['name'] = img
            img_dict['input'] = np.array(Image.open(os.path.join(base_path, img + '-INPUT.jpg')).convert('RGB'))
            img_dict['frame_mask'] = np.array(Image.open(os.path.join(base_path, img + '-INPUT-MASK.png')).convert('L'))
            img_dict['gt'] = np.array(Image.open(os.path.join(base_path, img + '-OUTPUT-GT.png')).convert('L'))
            img_dict['orig_size'] = img_dict['input'].shape
            images.append(img_dict)
        return images

    @staticmethod
    def convert_to_binary_mask(query_mask):
        # Initialize a binary mask with zeros (same height and width as query_mask)
        query_mask_binary = np.zeros(query_mask.shape[:2], dtype=np.uint8)

        # Set pixels matching the drawn_class to 1
        query_mask_binary[np.all(query_mask == [255, 255, 255], axis=-1)] = 1

        return query_mask_binary

    def load_frame(self, idx):
        if self.split == 'train':
            query_img, query_mask = self.images[0]['input'], self.images[0]['gt']
            query_name = 'train'
        else:
            list_idx, idx = self.global_to_local(idx)
            img = self.images[list_idx]
            query_img = img['input'][idx]
            query_mask = img['gt'][idx]
            query_name = f'{list_idx}_{idx}'

        augment = self.augmentations(image=query_img, mask=query_mask)
        query_img, query_mask = augment['image'], augment['mask']

        query_img = Image.fromarray(query_img)

        query_mask = query_mask / 255
        query_mask = torch.tensor(query_mask, dtype=torch.float)

        return query_img, query_mask, query_name
