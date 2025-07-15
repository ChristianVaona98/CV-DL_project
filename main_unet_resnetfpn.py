import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops
import time
import random
import argparse

from config import Config
from models.UNet import UNetFromScratch
from models.SemanticFPN import SemanticFPN
from utils.dataset_class import BrainTumorDataset
from utils.loss import DiceFocalLoss

def global_dice(preds, targets, threshold=0.5, smooth=1e-5):
    preds_bin = (preds > threshold).float()
    preds_flat = preds_bin.flatten()
    targets_flat = targets.flatten()
    intersection = (preds_flat * targets_flat).sum()
    return (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)

def global_precision_recall(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).float()
    preds_flat = preds_bin.flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()
    if np.sum(targets_flat) == 0:
        return 1.0 if np.sum(preds_flat) == 0 else 0.0, 1.0
    tp = np.sum((preds_flat == 1) & (targets_flat == 1))
    fp = np.sum((preds_flat == 1) & (targets_flat == 0))
    fn = np.sum((preds_flat == 0) & (targets_flat == 1))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return precision, recall

def lesion_sensitivity(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).squeeze(1).cpu().numpy()
    targets_bin = targets.squeeze(1).cpu().numpy()
    detected_lesions = 0
    total_lesions = 0
    for i in range(len(preds_bin)):
        target_label = label(targets_bin[i])
        pred_label = label(preds_bin[i])
        props = regionprops(target_label)
        total_lesions += len(props)
        for lesion in props:
            lesion_mask = (target_label == lesion.label)
            if np.any(pred_label[lesion_mask] > 0):
                detected_lesions += 1
    return detected_lesions / (total_lesions + 1e-7)

# Load data paths
def load_data_paths(config):
    image_paths = sorted([os.path.join(config.image_dir, f) for f in os.listdir(config.image_dir)])
    mask_paths = [] #sorted([os.path.join(config.mask_dir, f) for f in os.listdir(config.mask_dir)])
    
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(config.mask_dir, base_name)
  
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            print(f"Mask not found for image: {base_name}")
            image_paths.remove(img_path)
    
    print(f"Found {len(image_paths)} matched image-mask pairs")
    return image_paths, mask_paths

def split_dataset(image_paths, mask_paths, config):
    # Split dataset
    train_img, test_img, train_mask, test_mask = train_test_split(
        image_paths, mask_paths, test_size=config.test_ratio, random_state=config.seed
    )
    train_img, val_img, train_mask, val_mask = train_test_split(
        train_img, train_mask, test_size=config.val_ratio/(1-config.test_ratio),
        random_state=config.seed
    )

    print(f"Dataset % split: Train={config.train_ratio} Val={config.val_ratio} Test={config.test_ratio}")
    print(f"Dataset image split: Train={len(train_img)} Val={len(val_img)} Test={len(test_img)}")

    return (train_img, test_img, train_mask, test_mask,
            val_img, val_mask)
            
def apply_augmentation(config):
    # Create transforms
    train_transform = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    test_transform = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, test_transform

# Create datasets
def create_datasets(train_img, test_img, train_mask, test_mask,
            val_img, val_mask, train_transform, test_transform, config): 
    
    # Create datasets
    train_dataset = BrainTumorDataset(train_img, train_mask, train_transform, config)
    val_dataset = BrainTumorDataset(val_img, val_mask, test_transform, config)
    test_dataset = BrainTumorDataset(test_img, test_mask, test_transform, config)
    
    return train_dataset , val_dataset , test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    # Create dataloaders
    train_loader_unet = DataLoader(train_dataset, batch_size=config.unet_batch_size, shuffle=True, num_workers=4)
    val_loader_unet = DataLoader(val_dataset, batch_size=config.unet_batch_size, shuffle=False, num_workers=4)
    
    train_loader_semantic_fpn = DataLoader(train_dataset, batch_size=config.semantic_fpn_batch_size, shuffle=True, num_workers=4)
    val_loader_semantic_fpn = DataLoader(val_dataset, batch_size=config.semantic_fpn_batch_size, shuffle=False, num_workers=4)
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    return (train_loader_unet, val_loader_unet, 
            train_loader_semantic_fpn, val_loader_semantic_fpn,
            test_loader)

#%% Training and Evaluation Functions
def train_model(model, train_loader, val_loader, optimizer, criterion, model_name, epochs, config):
    model.to(config.device)
    results = {
        'train_loss': [], 'train_dice': [],
        'val_loss': [], 'val_dice': [], 
        'val_precision': [], 'val_recall': [], 
        'val_lesion_recall': [], 'lr_history': []
    }
    best_dice = 0.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    if model_name == "unet":
        model_path = config.unet_path
    else:
        model_path = config.semantic_fpn_path
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        all_preds, all_targets = [], []
        
        for images, masks in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}"):
            images, masks = images.to(config.device), masks.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_targets.append(masks.detach().cpu())
        
        # Calculate training metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        train_loss = epoch_train_loss / len(train_loader)
        train_dice = global_dice(all_preds, all_targets).item()
        
        # Validation
        val_loss, val_dice, val_prec, val_rec, val_lesion_rec = evaluate_model(model, val_loader, criterion, config)

        # Update scheduler based on validation dice
        scheduler.step(val_dice)
        
        # Save results
        results['train_loss'].append(train_loss)
        results['train_dice'].append(train_dice)
        results['val_loss'].append(val_loss)
        results['val_dice'].append(val_dice)
        results['val_precision'].append(val_prec)
        results['val_recall'].append(val_rec)
        results['val_lesion_recall'].append(val_lesion_rec)
        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        results['lr_history'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"-----------| Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print(f"-----------| Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | Lesion Rec: {val_lesion_rec:.4f}| "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best {model_name} model with Dice: {best_dice:.4f}")
    
    # Load best weights
    model.load_state_dict(torch.load(model_path))
    return model, results

def evaluate_model(model, dataloader, criterion, config):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(config.device), masks.to(config.device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_targets.append(masks.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    epoch_loss = running_loss / len(dataloader)
    dice = global_dice(all_preds, all_targets).item()
    precision, recall = global_precision_recall(all_preds, all_targets)
    lesion_recall = lesion_sensitivity(all_preds, all_targets)
    
    return epoch_loss, dice, precision, recall, lesion_recall

def save_results(results, model_name, config):
    with open(config.results_file, "a") as f:
        # Header with model name and timestamp
        f.write(f"\n\n{'='*60}\n")
        f.write(f"| {model_name.upper()} RESULTS\n")
        f.write(f"| Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        
        # Final metrics table
        f.write("Final Validation Metrics:\n")
        f.write("+-----------------+----------+----------+-----------------+\n")
        f.write("| Metric          | Value    | Metric   | Value           |\n")
        f.write("+-----------------+----------+----------+-----------------+\n")
        f.write(f"| Dice            | {results['val_dice'][-1]:.6f} | Precision | {results['val_precision'][-1]:.6f} |\n")
        f.write(f"| Recall          | {results['val_recall'][-1]:.6f} | LesionRec | {results['val_lesion_recall'][-1]:.6f} |\n")
        f.write("+-----------------+----------+----------+-----------------+\n\n")
        
        # Detailed epoch history
        f.write("Epoch-wise Performance:\n")
        f.write("+-------+------------+------------+------------+------------+------------+------------+\n")
        f.write("| Epoch | Train Loss | Train Dice | Val Loss   | Val Dice   | Val Prec   | Val Rec    |\n")
        f.write("+-------+------------+------------+------------+------------+------------+------------+\n")
        
        for i in range(len(results['train_loss'])):
            f.write(f"| {i+1:5d} "
                    f"| {results['train_loss'][i]:10.6f} "
                    f"| {results['train_dice'][i]:10.6f} "
                    f"| {results['val_loss'][i]:10.6f} "
                    f"| {results['val_dice'][i]:10.6f} "
                    f"| {results['val_precision'][i]:10.6f} "
                    f"| {results['val_recall'][i]:10.6f} |\n")
        
        f.write("+-------+------------+------------+------------+------------+------------+------------+\n")
        
        # Performance summary
        best_dice_idx = np.argmax(results['val_dice'])
        f.write("\nBest Performance Summary:\n")
        f.write(f"- Best Dice: {results['val_dice'][best_dice_idx]:.4f} at epoch {best_dice_idx+1}\n")
        f.write(f"- Corresponding Precision: {results['val_precision'][best_dice_idx]:.4f}\n")
        f.write(f"- Corresponding Recall: {results['val_recall'][best_dice_idx]:.4f}\n")
        f.write(f"- Lesion Recall: {results['val_lesion_recall'][best_dice_idx]:.4f}\n")
        
        # Add space for next model
        f.write("\n" + "="*60 + "\n\n")

def plot_training_curves(results, model_name, config):
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(results['train_loss'], label='Train Loss')
    plt.plot(results['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Dice curves
    plt.subplot(1, 2, 2)
    plt.plot(results['train_dice'], label='Train Dice')
    plt.plot(results['val_dice'], label='Val Dice')
    plt.title(f'{model_name} Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{config.curves_dir}/{model_name}_curves.png")
    plt.close()

def visualize_predictions(unet, semantic_fpn, test_loader, config, num_samples=5):
    # Get random samples
    all_indices = list(range(len(test_loader.dataset)))
    sample_indices = random.sample(all_indices, num_samples)
    sample_data = [test_loader.dataset[i] for i in sample_indices]
    images, masks = zip(*sample_data)
    images = torch.stack(images).to(config.device)
    masks = torch.stack(masks).cpu().numpy()
    
    # Get predictions
    with torch.no_grad():
        unet.eval(); semantic_fpn.eval()
        unet_preds = unet(images).cpu().numpy()
        semantic_preds = semantic_fpn(images).cpu().numpy()
    
    # Create overlay function
    def overlay(image, mask, color, alpha=0.5):
        image = image.copy()
        for c in range(3):
            image[:, :, c] = np.where(mask > 0.5,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
        return image
    
    # Prepare visualization
    plt.figure(figsize=(16, 4 * num_samples))
    for i in range(num_samples):
        # Process image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Get masks
        gt_mask = masks[i].squeeze()
        unet_mask = unet_preds[i].squeeze()
        semantic_mask = semantic_preds[i].squeeze()
        
        # Create overlays
        gt_overlay = overlay(img, gt_mask, [1, 0, 0])  # Red
        unet_overlay = overlay(img, unet_mask, [0, 1, 0])  # Green
        semantic_overlay = overlay(img, semantic_mask, [0, 0, 1])  # Blue
        
        # Plot
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(img)
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 2)
        plt.imshow(gt_overlay)
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(unet_overlay)
        plt.title('U-Net Prediction')
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(semantic_overlay)
        plt.title('SemanticFPN Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config.visualizations_dir}/comparison_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

def final_evaluation(unet_model, semantic_fpn_model, test_loader, criterion, config):
    # Final evaluation on test set
    print("\n==== Final Test Evaluation ====")
    with open(config.results_file, "a") as f:
        f.write("\n\n===== Final Test Results =====\n")
    
    for name, model in [("U-Net", unet_model), ("SemanticFPN", semantic_fpn_model)]:
        test_loss, test_dice, test_prec, test_rec, test_lesion_rec = evaluate_model(
            model, test_loader, criterion, config)
        
        print(f"\n{name} Test Results:")
        print(f"Dice: {test_dice:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | Lesion Recall: {test_lesion_rec:.4f}")
        
        with open(config.results_file, "a") as f:
            f.write(f"\n{name}:\n")
            f.write(f"  Dice: {test_dice:.4f}\n")
            f.write(f"  Precision: {test_prec:.4f}\n")
            f.write(f"  Recall: {test_rec:.4f}\n")
            f.write(f"  Lesion Recall: {test_lesion_rec:.4f}\n")


        time_results = measure_inference_time(model, name, test_loader, config)
        # Save results
        save_inference_results(time_results, config)

def measure_inference_time(model, name, test_loader, config, num_tests=100):
    results = {}
    device = config.device
    
    # Prepare test subset
    total_images = min(num_tests, len(test_loader.dataset))
    test_subset = torch.utils.data.Subset(test_loader.dataset, range(total_images))
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()
    
    # Warm-up GPU
    print(f"Warming up GPU for {name}...")
    warmup_tensor = torch.randn(1, 3, config.img_size, config.img_size).to(device)
    for _ in range(10):
        _ = model(warmup_tensor)
    
    # Measure inference
    print(f"Measuring inference time for {name} on {total_images} images...")
    times = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, total=total_images):
            images = images.to(device)
            
            # Synchronize and start timer
            if config.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Forward pass
            _ = model(images)
            
            # Synchronize and record
            if config.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
    
    # Calculate metrics
    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    fps = 1000 / avg_time if avg_time > 0 else float('inf')
    
    results[name] = {
        'avg_time_ms': avg_time,
        'fps': fps,
        'total_time': np.sum(times),
        'num_images': total_images
    }

    return results

def save_inference_results(time_results, config):
    """Saves inference time results in a formatted table"""
    with open(config.results_file, "a") as f:
        f.write("\n\n" + "="*80 + "\n")
        f.write("INFERENCE TIME COMPARISON\n")
        f.write(f"Tested on {next(iter(time_results.values()))['num_images']} images\n")
        f.write("="*80 + "\n\n")
        
        f.write("+--------------+----------------+----------------+----------------+\n")
        f.write("| Model        | Avg Time (ms)  | FPS            | Total Time (s) |\n")
        f.write("+--------------+----------------+----------------+----------------+\n")
        
        for model_name, metrics in time_results.items():
            f.write(f"| {model_name:12} | {metrics['avg_time_ms']:14.3f} | "
                    f"{metrics['fps']:14.2f} | {metrics['total_time']:14.3f} |\n")
        
        f.write("+--------------+----------------+----------------+----------------+\n\n")
        
        # Find fastest model
        fastest = min(time_results.items(), key=lambda x: x[1]['avg_time_ms'])
        f.write(f"Fastest model: {fastest[0]} ({fastest[1]['avg_time_ms']:.2f} ms/image)\n")

#%% Main Execution
def run_experiment(seed, output_dir):
    # Initialize configuration with seed and output directory
    config = Config(seed, output_dir)

    # Initialize results file
    with open(config.results_file, "w") as f:
        f.write(f"Brain Tumor Segmentation Results\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image Size: {config.img_size}\n")
        f.write(f"Epochs: {config.epochs}\n\n")
    
    # Load dataset paths
    image_paths, mask_paths = load_data_paths(config)
    
    train_img, test_img, train_mask, test_mask, val_img, val_mask  = split_dataset(image_paths, mask_paths, config)
    
    train_transform, test_transform  = apply_augmentation(config)

    train_dataset, val_dataset, test_dataset  = create_datasets(train_img, test_img, train_mask, test_mask,
            val_img, val_mask, train_transform, test_transform, config)
    
    (train_loader_unet, val_loader_unet, 
    train_loader_semantic_fpn, val_loader_semantic_fpn, 
    test_loader) = create_dataloaders(train_dataset, val_dataset, test_dataset, config)
    
    # Initialize models
    unet = UNetFromScratch()
    semantic_fpn = SemanticFPN()
    
    # Loss function
    criterion = DiceFocalLoss()
    
    # Train or load UNET
    if config.load_unet:
        print("\nLoading pre-trained U-Net...")
        unet.load_state_dict(torch.load(config.unet_path))
        unet.to(config.device)
        unet_results = None
    else:
        print("\nTraining U-Net...")
        optimizer = torch.optim.Adam(unet.parameters(), lr=config.lr, weight_decay=1e-5)
        unet, unet_results = train_model(unet, train_loader_unet, val_loader_unet, 
                                        optimizer, criterion, "unet", config.epochs, config)
        if unet_results:
            save_results(unet_results, "U-Net", config)
            plot_training_curves(unet_results, "U-Net", config)
    
    # Train or load SemanticFPN
    if config.load_semantic_fpn:
        print("\nLoading pre-trained SemanticFPN...")
        semantic_fpn.load_state_dict(torch.load(config.semantic_fpn_path))
        semantic_fpn.to(config.device)
        semantic_results = None
    else:
        print("\nTraining SemanticFPN...")
        optimizer = torch.optim.Adam(semantic_fpn.parameters(), lr=config.lr, weight_decay=1e-5)
        semantic_fpn, semantic_results = train_model(semantic_fpn, train_loader_semantic_fpn, val_loader_semantic_fpn,
                                               optimizer, criterion, "semantic_fpn", config.epochs, config)
        if semantic_results:
            save_results(semantic_results, "SemanticFPN", config)
            plot_training_curves(semantic_results, "SemanticFPN", config)
    
    final_evaluation(unet, semantic_fpn, test_loader, criterion, config)
    
    # Visualize predictions
    visualize_predictions(unet, semantic_fpn, test_loader, config)
    
    print("\nTraining complete! Results saved to", config.results_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    run_experiment(seed=args.seed, output_dir=args.output_dir)