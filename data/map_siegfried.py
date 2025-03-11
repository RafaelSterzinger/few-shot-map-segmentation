import os
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import albumentations as A


class DatasetSiegfried(Dataset):
    def __init__(self, datapath, transform, split, nshots, is_unet=False):
        print('Loading data from %s' % datapath)
        self.split = split
        self.benchmark = 'maps_siegfried'
        self.base_path = datapath
        self.transform = transform
        self.is_unet = is_unet

        self.img_metadata = self.build_img_metadata(self.split)

        if split == 'train':
            if type(nshots) is str:
                if '.' in nshots:
                    nshots = float(nshots)
                else:
                    nshots = int(nshots)
            
            if type(nshots) is float:
                self.img_metadata = random.sample(
                    self.img_metadata, round(len(self.img_metadata) * nshots))
                print(f'Training with {nshots*100} percent')
            elif type(nshots) is int:
                self.img_metadata = random.sample(self.img_metadata, nshots)
                print(f'Training with {nshots} samples')

            self.augmentations = A.Compose([
                    A.D4(),
                ])
        else:
            self.augmentations = A.Compose([A.NoOp()])

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_img, query_mask, query_name = self.load_frame(idx)

        #query_mask = query_mask.float()
        if self.is_unet:
            query_img = self.transform(query_img)
        else:
            query_img = self.transform(query_img)['pixel_values'][0]

        batch = {'img': query_img,
                 'mask': query_mask,
                 'name': query_name,
                 }

        return batch

    def build_img_metadata(self, split):
        def read_metadata(split):
            data = os.listdir(os.path.join(self.base_path, split))
            fold_n_metadata = [d.split('/')[-1].split('.')[0] for d in data]
            return fold_n_metadata

        img_metadata = read_metadata(split)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    @staticmethod
    def convert_to_binary_mask(query_mask):
        # Initialize a binary mask with zeros (same height and width as query_mask)
        query_mask_binary = np.zeros(query_mask.shape[:2], dtype=np.uint8)

        # Set pixels matching the drawn_class to 1
        query_mask_binary[np.all(query_mask == [255, 255, 255], axis=-1)] = 1

        return query_mask_binary

    def load_frame(self, idx):
        idx = idx % self.img_metadata.__len__()
        query_name = self.img_metadata[idx]

        query_img = np.array(Image.open(os.path.join(
            self.base_path, self.split, query_name + '.tif')).convert('RGB'))
        query_mask = np.array(Image.open(os.path.join(
            self.base_path, 'annotation', self.split, query_name + '.tif')).convert('RGB'))

        augment = self.augmentations(image=query_img, mask=query_mask)
        query_img, query_mask = augment['image'], augment['mask']

        query_img = Image.fromarray(query_img)

#        # Get original size
#        width, height = query_img.size  
#        
#        # Resize to double the dimensions
#        query_img = query_img.resize((width * 2, height * 2))

        query_mask = DatasetSiegfried.convert_to_binary_mask(query_mask)
        query_mask = torch.tensor(query_mask, dtype=torch.float)

        return query_img, query_mask, query_name
