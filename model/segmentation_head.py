from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import torch.nn as nn
import torch

INPUT_CHANNELS = {
    'radio' : 1280,
    'dino' : 1024,
    'sam' : 256,
    'apple' : 25
    
}

class SegmentationHead(nn.Module):
    def __init__(self, encoder, enc_name):
        super().__init__()
        self.encoder = encoder
        self.enc_name = enc_name
        self.seg_head = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS[enc_name], 1, kernel_size=1, padding=1),
        )
        self.upsample = nn.Upsample(224, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        features = self.encode_batch(x)
        logits = self.decode_batch(features).squeeze(1)
        return torch.sigmoid(logits)
    
    def decode_batch(self, x):
        x = self.seg_head(x)
        x = self.upsample(x)
        return x

    def encode_batch(self, x):
        if self.enc_name == 'radio':
            features = self.encoder(x).features
        elif self.enc_name == 'dino':
            features = self.encoder(x).last_hidden_state[:,1:,:]
        else:
            features = self.encoder(x).last_hidden_state
            B,D,H,W = features.shape
            features = features.view(B,D,H*W).permute(0,2,1)

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

     # Plot the PCA visualization
     plt.figure(figsize=(6, 6))
     plt.imshow(transformed_img)
     plt.axis('off')
     plt.title('PCA Visualization of Feature Map')
     plt.show()

# https://github.com/facebookresearch/MaskFormer/tree/main/mask_former for TTA