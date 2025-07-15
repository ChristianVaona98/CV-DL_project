import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, config=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.config = config
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
            mask = (mask > 128).float().unsqueeze(0) # (B, 1, H, W)
        else:
            img = cv2.resize(img, (self.config.img_size, self.config.img_size))
            mask = cv2.resize(mask, (self.config.img_size, self.config.img_size))
            mask = (mask > 128).astype(np.float32)
            mask = torch.tensor(mask).unsqueeze(0).float()
            
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img = torch.tensor(img).permute(2, 0, 1).float()
        
        return img, mask