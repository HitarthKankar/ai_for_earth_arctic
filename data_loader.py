"""
Sentinel-2 Data Loader for PyTorch

This module provides a PyTorch Dataset for loading Sentinel-2 satellite imagery
and corresponding lake occurrence labels from the Alaska Lake and Pond Occurrence
Dataset (ALPOD).

The data consists of:
- 100 image grids covering a 100km x 100km region in Alaska
- Each grid contains 10 Sentinel-2 bands at 10m resolution
- Spatial dimensions: 1098 x 1097 pixels
- Corresponding pixel-level lake occurrence labels (0-100 percentage)

Students are responsible for:
1. Creating train/val/test splits
2. Performing any necessary preprocessing
3. Creating patches if needed for their models
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from pathlib import Path


class Sentinel2Dataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 satellite imagery and lake occurrence labels.
    
    Loads full image grids and their corresponding labels. Students can then
    implement their own patching, augmentation, and preprocessing strategies.
    
    Args:
        image_dir (str): Path to directory containing image numpy arrays
        label_dir (str): Path to directory containing label numpy arrays
        image_files (list, optional): List of specific image filenames to load.
                                     If None, loads all files in image_dir.
        return_filenames (bool): If True, returns (image, label, filename) tuple
        device (str): Device to load data to ('cpu' or 'cuda')
    
    Returns:
        Tuple of (image, label) or (image, label, filename) if return_filenames=True
        - image: torch.Tensor of shape [10, 1098, 1097] - 10 Sentinel-2 bands
        - label: torch.Tensor of shape [1098, 1097] - lake occurrence percentage
        - filename: str (optional) - original numpy filename
    """
    
    def __init__(self, image_dir, label_dir, image_files=None, 
                 return_filenames=False, device='cpu'):
        """Initialize the dataset."""
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.return_filenames = return_filenames
        self.device = device
        
        # Get list of image files
        if image_files is None:
            self.image_files = sorted([f for f in os.listdir(image_dir) 
                                      if f.endswith('.npy')])
        else:
            self.image_files = image_files
        
        if len(self.image_files) == 0:
            raise ValueError(f"No .npy files found in {image_dir}")
        
        print(f"Loaded {len(self.image_files)} image grids from {image_dir}")
    
    def __len__(self):
        """Return the total number of grids in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Load a single image-label pair.
        
        Args:
            idx (int): Index of the grid to load
        
        Returns:
            Tuple of (image, label) or (image, label, filename)
        """
        image_file = self.image_files[idx]
        
        # Extract grid identifier from filename (e.g., "0_0" from "S2_IMAGE_0_0.npy")
        grid_id = "_".join(image_file.split("_")[2:4])
        label_file = f"S2_LABELS_{grid_id}.npy"
        
        # Load image and label
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, label_file)
        
        image = np.load(image_path).astype(np.float32)  # [10, 1098, 1097]
        label = np.load(label_path).astype(np.float32)  # [1098, 1097]
        
        # Convert to torch tensors
        image = torch.from_numpy(image).to(self.device)
        label = torch.from_numpy(label).to(self.device)
        
        if self.return_filenames:
            return image, label, image_file
        return image, label
    
    def get_grid_info(self, idx):
        """
        Get information about a specific grid.
        
        Args:
            idx (int): Index of the grid
        
        Returns:
            dict: Information about the grid including filename and coordinates
        """
        image_file = self.image_files[idx]
        grid_id = "_".join(image_file.split("_")[2:4])
        parts = grid_id.split("_")
        
        return {
            'filename': image_file,
            'grid_id': grid_id,
            'row': int(parts[0]),
            'column': int(parts[1])
        }
    
    def get_grid_ids(self):
        """
        Get all grid identifiers in the dataset.
        
        Returns:
            list: Grid identifiers (e.g., ["0_0", "0_1", ..., "9_9"])
        """
        grid_ids = []
        for image_file in self.image_files:
            grid_id = "_".join(image_file.split("_")[2:4])
            grid_ids.append(grid_id)
        return grid_ids


class SplitManager:
    """
    Utility class to help students create and manage train/val/test splits.
    
    This class makes it easy to organize the 100 grids into different splits
    and create corresponding Dataset instances.
    """
    
    def __init__(self, image_dir, label_dir):
        """Initialize with data directories."""
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        # Get all available files
        all_files = sorted([f for f in os.listdir(image_dir) 
                           if f.endswith('.npy')])
        self.all_grid_ids = [f.split("_")[2:4] for f in all_files]
        self.all_image_files = all_files
    
    def create_split(self, grid_indices):
        """
        Create a dataset with a specific subset of grids.
        
        Args:
            grid_indices (list): List of indices (0-99) to include in split
        
        Returns:
            Sentinel2Dataset: Dataset containing only specified grids
        """
        selected_files = [self.all_image_files[i] for i in grid_indices]
        return Sentinel2Dataset(self.image_dir, self.label_dir, 
                               image_files=selected_files)
    
    def random_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Create random train/val/test splits.
        
        Args:
            train_ratio (float): Proportion for training (default 0.7)
            val_ratio (float): Proportion for validation (default 0.15)
            test_ratio (float): Proportion for testing (default 0.15)
        
        Returns:
            dict: Contains 'train', 'val', 'test' Sentinel2Dataset objects
        """
        n = len(self.all_image_files)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        
        train_idx = indices[:train_n]
        val_idx = indices[train_n:train_n + val_n]
        test_idx = indices[train_n + val_n:]
        
        return {
            'train': self.create_split(train_idx),
            'val': self.create_split(val_idx),
            'test': self.create_split(test_idx)
        }
    
    def grouped_split(self, train_end=70, val_end=85):
        """
        Create train/val/test splits by keeping grids grouped.
        
        Grids 0-69 → train, 70-84 → val, 85-99 → test
        (Can be customized with train_end and val_end parameters)
        
        Args:
            train_end (int): Index where training set ends (default 70)
            val_end (int): Index where validation set ends (default 85)
        
        Returns:
            dict: Contains 'train', 'val', 'test' Sentinel2Dataset objects
        """
        return {
            'train': self.create_split(list(range(0, train_end))),
            'val': self.create_split(list(range(train_end, val_end))),
            'test': self.create_split(list(range(val_end, len(self.all_image_files))))
        }


class BandInfo:
    """
    Utility class providing information about Sentinel-2 bands.
    
    Useful for visualizing specific bands or understanding their properties.
    """
    
    BANDS_INFO = {
        0: {'name': 'B2', 'wavelength': 'Blue (496.6 nm)', 'resolution': '10m'},
        1: {'name': 'B3', 'wavelength': 'Green (560 nm)', 'resolution': '10m'},
        2: {'name': 'B4', 'wavelength': 'Red (664.5 nm)', 'resolution': '10m'},
        3: {'name': 'B5', 'wavelength': 'Red Edge 1 (703.9 nm)', 'resolution': '20m'},
        4: {'name': 'B6', 'wavelength': 'Red Edge 2 (740.2 nm)', 'resolution': '20m'},
        5: {'name': 'B7', 'wavelength': 'Red Edge 3 (782.8 nm)', 'resolution': '20m'},
        6: {'name': 'B8', 'wavelength': 'NIR (835.1 nm)', 'resolution': '10m'},
        7: {'name': 'B8a', 'wavelength': 'Red Edge 4 (864.8 nm)', 'resolution': '20m'},
        8: {'name': 'B11', 'wavelength': 'SWIR 1 (1613.7 nm)', 'resolution': '20m'},
        9: {'name': 'B12', 'wavelength': 'SWIR 2 (2202.4 nm)', 'resolution': '20m'},
    }
    
    @classmethod
    def get_band_info(cls, band_index):
        """Get information about a specific band."""
        return cls.BANDS_INFO.get(band_index)
    
    @classmethod
    def print_all_bands(cls):
        """Print information about all bands."""
        print("\nSentinel-2 Bands:")
        print("-" * 80)
        for idx, info in cls.BANDS_INFO.items():
            print(f"Band {idx}: {info['name']:6} | {info['wavelength']:30} | {info['resolution']}")
        print("-" * 80)
