import numpy as np
from sklearn.decomposition import PCA
from skimage.transform import resize
import cv2
import torch.nn as nn
import torch

INPUT_CHANNELS = {
    'radio_l' : 1024,
    'radio_h' : 1280,
    'dino' : 1024,
    'sam' : 256,
    'apple' : 1024
    
}

class SegmentationHead(nn.Module):
    def __init__(self, encoder, enc_name):
        super().__init__()
        self.encoder = encoder
        self.enc_name = enc_name
        self.seg_head = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS[enc_name], 1, kernel_size=1),
        )
        self.upsample = nn.Upsample(224*2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        features = self.encode_batch(x)
        logits = self.decode_batch(features).squeeze(1)
        return torch.sigmoid(logits)
    
    def decode_batch(self, x):
        x = self.upsample(x)
        x = self.seg_head(x)
        return x

    def encode_batch(self, x):
        if 'radio' in self.enc_name:
            features = self.encoder(x).features
        elif self.enc_name == 'dino':
            features = self.encoder(x).last_hidden_state[:,1:,:]
        elif self.enc_name == 'sam':
            features = self.encoder(x).last_hidden_state
            B,D,H,W = features.shape
            features = features.view(B,D,H*W).permute(0,2,1)
        elif self.enc_name == 'apple':
            features = self.encoder(x).last_hidden_state
        else:
            raise NotImplementedError

        B,HW_flattend,D = features.shape
        HW = int(HW_flattend**0.5)
        assert HW**2 == HW_flattend
        features = features.permute(0,2,1).view(B,D,HW,HW)
        return features
    
    def visualize_pca(self, x):
     """
     Visualizes the first three principal components of the given features.

     Parameters:
     features (torch.Tensor): Tensor of shape (B, D, HW, HW)
     """
     features = self.encode_batch(x)

     B, D, H, W = features.shape

     # Select the first batch for visualization
     feature_map = features[0].reshape(D, H * W).cpu().numpy()  # Shape (D, HW*HW)

     # Apply PCA to reduce dimensions to 3
     pca = PCA(n_components=3)
     transformed = pca.fit_transform(feature_map.T)  # Shape (HW*HW, 3)

     # Reshape back to (H, W, 3) for visualization
     transformed_img = transformed.reshape(H, W, 3)

     # Normalize to [0,1] for display
     transformed_img = (transformed_img - transformed_img.min()) / (transformed_img.max() - transformed_img.min())
     resized_image = resize(transformed_img, (224, 224, 3),order=3)
     resized_image_uint8 = (resized_image * 255).astype(np.uint8)
     cv2.imwrite(f'pca_{self.enc_name}.png', resized_image_uint8)
     exit()


# https://github.com/facebookresearch/MaskFormer/tree/main/mask_former for TTA