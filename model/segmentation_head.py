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
            nn.Conv2d(INPUT_CHANNELS[enc_name], 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.upsample = nn.Upsample(224, mode='bilinear', align_corners=True)
    
    def forward(self, x):
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

        logits = self.seg_head(features)
        logits = self.upsample(logits).squeeze(1)
        return torch.sigmoid(logits)