"""
Sentinel-2 Data Loader Tests and Examples

This script demonstrates how to use the Sentinel-2 data loader for:
1. Loading full grids
2. Creating train/val/test splits
3. Visualizing data
4. Analyzing water coverage
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
from torch.utils.data import DataLoader

from data_loader import Sentinel2Dataset, SplitManager, BandInfo


def load_data(image_dir, label_dir):
    """
    Basic example: Load all data.
    
    Args:
        image_dir (str): Path to image directory
        label_dir (str): Path to label directory
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Loading All Data")
    print("="*80)
    
    dataset = Sentinel2Dataset(image_dir, label_dir, return_filenames=True)
    
    # Access first grid
    image, label, filename = dataset[0]
    print(f"\nLoaded: {filename}")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Image dtype: {image.dtype}, Label dtype: {label.dtype}")
    
    return dataset


def create_splits(image_dir, label_dir):
    """
    Example: Create train/val/test splits.
    
    Args:
        image_dir (str): Path to image directory
        label_dir (str): Path to label directory
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Creating Train/Val/Test Splits")
    print("="*80)
    
    split_manager = SplitManager(image_dir, label_dir)
    
    # Option 1: Grouped split (sequential)
    print("\nGrouped Split (Sequential):")
    splits = split_manager.grouped_split(train_end=70, val_end=85)
    print(f"  Train: {len(splits['train'])} grids")
    print(f"  Val: {len(splits['val'])} grids")
    print(f"  Test: {len(splits['test'])} grids")
    
    # Option 2: Random split
    print("\nRandom Split:")
    splits_random = split_manager.random_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"  Train: {len(splits_random['train'])} grids")
    print(f"  Val: {len(splits_random['val'])} grids")
    print(f"  Test: {len(splits_random['test'])} grids")
    
    return splits


def visualize_grid(image_dir, label_dir, grid_index=0):
    """
    Example: Visualize a grid and its label.
    
    Args:
        image_dir (str): Path to image directory
        label_dir (str): Path to label directory
        grid_index (int): Index of grid to visualize
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Visualizing Grids")
    print("="*80)
    
    dataset = Sentinel2Dataset(image_dir, label_dir, return_filenames=True)
    
    image, label, filename = dataset[grid_index]
    grid_info = dataset.get_grid_info(grid_index)
    
    print(f"\nVisualizing: {filename}")
    print(f"Grid position: Row {grid_info['row']}, Column {grid_info['column']}")
    
    # Convert to numpy for visualization
    image_np = image.numpy()
    label_np = label.numpy()
    
    # Create RGB visualization (Red, Green, Blue bands)
    rgb = np.stack([image_np[2], image_np[1], image_np[0]], axis=-1)  # R, G, B
    
    # Normalize RGB to 0-1 for visualization
    rgb = np.clip(rgb / rgb.max(), 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB image
    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB Composite\n{filename}')
    axes[0].axis('off')
    
    # Label (lake occurrence)
    im = axes[1].imshow(label_np, cmap='Blues', vmin=0, vmax=100)
    axes[1].set_title('Lake Occurrence (%)')
    plt.colorbar(im, ax=axes[1])
    
    # NIR band (good for water detection)
    nir = image_np[6]  # B8 - NIR band
    axes[2].imshow(nir, cmap='gray')
    axes[2].set_title('NIR Band (B8)')
    
    plt.tight_layout()
    plt.savefig('sentinel2_visualization.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved as 'sentinel2_visualization.png'")
    plt.show()


def analyze_water_coverage(image_dir, label_dir, threshold=25):
    """
    Example: Analyze water coverage across grids.
    
    Similar to the tutorial notebook example, but integrated with the dataloader.
    
    Args:
        image_dir (str): Path to image directory
        label_dir (str): Path to label directory
        threshold (float): Percentage threshold for water detection
    """
    print("\n" + "="*80)
    print(f"EXAMPLE 4: Water Coverage Analysis (threshold={threshold}%)")
    print("="*80)
    
    dataset = Sentinel2Dataset(image_dir, label_dir, return_filenames=True)
    
    water_coverage = {}
    
    for idx in range(len(dataset)):
        image, label, filename = dataset[idx]
        grid_info = dataset.get_grid_info(idx)
        
        # Binarize label at threshold
        binarized = (label >= threshold).float()
        coverage = binarized.sum().item() / binarized.numel()
        
        grid_id = grid_info['grid_id']
        water_coverage[grid_id] = coverage
    
    # Sort by coverage
    sorted_coverage = dict(sorted(water_coverage.items(), 
                                  key=lambda x: x[1], reverse=True))
    
    print(f"\nWater Coverage Statistics (threshold={threshold}%):")
    print("-" * 50)
    print(f"{'Grid ID':<15} {'Water Coverage (%)':<20}")
    print("-" * 50)
    
    for grid_id, coverage in list(sorted_coverage.items())[:20]:  # Show top 20
        print(f"{grid_id:<15} {coverage*100:>18.2f}%")
    
    print("\n...")
    print(f"\nAverage water coverage: {np.mean(list(water_coverage.values()))*100:.2f}%")
    print(f"Min water coverage: {min(water_coverage.values())*100:.2f}%")
    print(f"Max water coverage: {max(water_coverage.values())*100:.2f}%")


def create_dataloader(image_dir, label_dir, batch_size=2):
    """
    Example: Create PyTorch DataLoaders.
    
    Args:
        image_dir (str): Path to image directory
        label_dir (str): Path to label directory
        batch_size (int): Batch size for DataLoader
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Creating PyTorch DataLoaders")
    print("="*80)
    
    # Create splits
    split_manager = SplitManager(image_dir, label_dir)
    splits = split_manager.grouped_split()
    
    # Create DataLoaders
    train_loader = DataLoader(
        splits['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Adjust based on your system
        pin_memory=False
    )
    
    val_loader = DataLoader(
        splits['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        splits['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Show a sample batch
    print(f"\nSample batch from training set:")
    for images, labels in train_loader:
        print(f"  Image batch shape: {images.shape}")
        print(f"  Label batch shape: {labels.shape}")
        break


def print_band_info():
    """Print information about Sentinel-2 bands."""
    print("\n" + "="*80)
    print("Sentinel-2 Band Information")
    print("="*80)
    BandInfo.print_all_bands()


if __name__ == "__main__":
    # Update these paths to your local data directories
    IMAGE_DIR = "data/S2_IMAGE_NUMPY"
    LABEL_DIR = "data/S2_LABELS_NUMPY"
    
    # Check if data directories exist
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(LABEL_DIR):
        print(f"\nError: Data directories not found!")
        print(f"  Expected: {IMAGE_DIR}")
        print(f"  Expected: {LABEL_DIR}")
        print("\nPlease download the data from:")
        print("https://drive.google.com/drive/folders/1OM6N5ysD39TV9YGIVO-P3-c5AfUQnQ3m?usp=sharing")
        print("\nAnd extract to match the directory structure above.")
        exit(1)
    
    # Run examples
    print("\n" + "="*80)
    print("SENTINEL-2 DATA LOADER TESTS AND EXAMPLES")
    print("="*80)
    
    # Print band info
    print_band_info()
    
    # Example 1: Load data
    dataset = load_data(IMAGE_DIR, LABEL_DIR)
    
    # Example 2: Create splits
    splits = create_splits(IMAGE_DIR, LABEL_DIR)
    
    # Example 3: Visualize
    visualize_grid(IMAGE_DIR, LABEL_DIR, grid_index=0)
    
    # Example 4: Analyze water coverage
    analyze_water_coverage(IMAGE_DIR, LABEL_DIR, threshold=25)
    
    # Example 5: Create DataLoaders
    create_dataloader(IMAGE_DIR, LABEL_DIR, batch_size=2)
    
    print("\n" + "="*80)
    print("Tests completed!")
    print("="*80)
