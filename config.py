import numpy as np
import os
import torch
import random

class Config:
    # Data paths
    data_path = './brain-tumor-segmentation'
    image_dir = os.path.join(data_path, "images")
    mask_dir = os.path.join(data_path, "masks")
    
    # Model parameters
    img_size = 256
    unet_batch_size = 32
    semantic_fpn_batch_size = 32
    num_classes = 1  # Binary segmentation
    
    # Training
    lr = 1e-4
    epochs = 80
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load_unet = False  # Set to False to train UNet from scratch
    unet_path = "best_unet.pth"  
    load_semantic_fpn = False
    semantic_fpn_path = "best_semantic_fpn.pth"
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Output
    results_file = "results.txt"
    curves_dir = "training_curves"
    visualizations_dir = "predictions"

    def __init__(self, seed, output_dir):
        # Set seeds first for reproducibility
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False   
    
        # Create directories
        self.results_file = os.path.join(output_dir, self.results_file)
        self.curves_dir = os.path.join(output_dir, self.curves_dir)
        self.visualizations_dir = os.path.join(output_dir, self.visualizations_dir)

        os.makedirs(self.curves_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

        self.unet_path = os.path.join(output_dir, self.unet_path)
        self.semantic_fpn_path = os.path.join(output_dir, self.semantic_fpn_path)

        # Print config
        print("\nExperiment Configuration:")
        print(f"  Torch Seed: {seed}")
        print(f"  Output Directory: {output_dir}")
        for k, v in self.__dict__.items():
            if k not in ['seed', 'output_dir']:
                print(f"  {k}: {v}")