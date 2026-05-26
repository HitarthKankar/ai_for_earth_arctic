# Sentinel-2 Data Loader for PyTorch

A PyTorch data loader for Sentinel-2 satellite imagery and Alaska Lake and Pond Occurrence (ALPOD) dataset labels. Designed for machine learning research on water body detection across Alaska.

## Overview

This project provides tools for loading and processing Sentinel-2 satellite data combined with lake occurrence labels for the Arctic region. The dataset consists of:

- **100 image grids** covering a 100km × 100km region in Alaska (tile T03VWJ)
- **10 Sentinel-2 bands** at 10m resolution (Blue, Green, Red, Red Edge 1-4, NIR, SWIR 1-2)
- **1098 × 1097 spatial dimensions** per grid
- **Pixel-level lake occurrence labels** (0-100 percentage from ALPOD dataset)
- **July 2019 cloud-free composite**

## Features

- ✅ **Full Grid Loading**: Load complete image grids for maximum spatial context
- ✅ **Flexible Splits**: Easy train/val/test split creation (grouped or random)
- ✅ **Raw Labels**: Preserves original lake occurrence percentages (0-100)
- ✅ **PyTorch Integration**: Native PyTorch Dataset and DataLoader support
- ✅ **Grid Information**: Track grid coordinates and filenames
- ✅ **Band Information**: Easy access to Sentinel-2 band metadata

## Repository Structure

```
s2_dataloader/
├── data_loader.py              # Core PyTorch Dataset classes
├── test_dataloader.py          # Tests, examples, and visualization
├── config.yaml                 # Configuration and band information
├── utils/
│   ├── __init__.py
│   ├── YParams.py             # YAML parameter loader
│   └── BiMap.py               # Bidirectional mapping utility
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (CPU or GPU)
- Required packages: numpy, matplotlib, pyyaml

### Setup

1. **Clone the repository** (or initialize git):
```bash
cd s2_dataloader
git init
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download data**: Use [this link](https://drive.google.com/drive/folders/1OM6N5ysD39TV9YGIVO-P3-c5AfUQnQ3m?usp=sharing) to download the data.
   - Extract to create the directory structure:
     ```
     s2_dataloader/
     ├── data/
     │   ├── S2_IMAGE_NUMPY/    # 100 image files (.npy)
     │   └── S2_LABELS_NUMPY/   # 100 label files (.npy)
     ├── data_loader.py
     ...
     ```

4. **Update config paths** if needed in `config.yaml`:
   ```yaml
   DATA_CONFIG:
     IMAGE_DIR: "data/S2_IMAGE_NUMPY"
     LABEL_DIR: "data/S2_LABELS_NUMPY"
   ```

## Quick Start

See [data_loader.py](data_loader.py) for the complete API documentation. Here's what you can do:

**Load data:**
- `Sentinel2Dataset`: Main dataset class - loads full grids
- `SplitManager`: Create train/val/test splits
- `BandInfo`: Access Sentinel-2 band metadata

**Test and explore:**
```bash
python test_dataloader.py
```

This runs 5 examples demonstrating:
1. Loading all data
2. Creating train/val/test splits (grouped or random)
3. Visualizing grids and labels
4. Analyzing water coverage across grids
5. Creating PyTorch DataLoaders

**Visualization Example:**
```python
import matplotlib.pyplot as plt
from data_loader import Sentinel2Dataset
import numpy as np

# Load dataset
dataset = Sentinel2Dataset("data/S2_IMAGE_NUMPY", "data/S2_LABELS_NUMPY")

# Get first grid
image, label = dataset[0]

# Create RGB visualization (bands 2, 1, 0 = R, G, B)
rgb = np.stack([image[2], image[1], image[0]], axis=-1)
rgb = np.clip(rgb / rgb.max(), 0, 1)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(rgb)
axes[0].set_title('RGB Composite')
axes[1].imshow(label.numpy(), cmap='Blues', vmin=0, vmax=100)
axes[1].set_title('Lake Occurrence (%)')
plt.show()
```

## Sentinel-2 Bands

The 10 bands included in the dataset:

| Index | Band Name | Wavelength | Resolution | Description |
|-------|-----------|-----------|------------|-------------|
| 0 | B2 | Blue (496.6 nm) | 10m | Blue light |
| 1 | B3 | Green (560 nm) | 10m | Green light |
| 2 | B4 | Red (664.5 nm) | 10m | Red light |
| 3 | B5 | Red Edge 1 (703.9 nm) | 20m | Red edge |
| 4 | B6 | Red Edge 2 (740.2 nm) | 20m | Red edge |
| 5 | B7 | Red Edge 3 (782.8 nm) | 20m | Red edge |
| 6 | B8 | NIR (835.1 nm) | 10m | Near-infrared (excellent for water) |
| 7 | B8a | Red Edge 4 (864.8 nm) | 20m | Red edge |
| 8 | B11 | SWIR 1 (1613.7 nm) | 20m | Shortwave infrared |
| 9 | B12 | SWIR 2 (2202.4 nm) | 20m | Shortwave infrared |

**Note**: 20m resolution bands are resampled to 10m in these arrays, resulting in repeated values in 2×2 pixel blocks.

## Label Information

- **Source**: Alaska Lake and Pond Occurrence Dataset (ALPOD)
- **Values**: 0-100 (percentage of time period with open water occurrence)
- **Recommendation**: Threshold at 25% for binary classification (as in ALPOD paper)

Example binarization:
```python
binarized_labels = (labels >= 25).float()
```

## Dataset Characteristics

- **Region**: Alaska, Arctic region (tile T03VWJ in Sentinel-2 tiling system)
- **Spatial Coverage**: ~100km × 100km
- **Temporal Coverage**: July 2019 (cloud-free composite)
- **Grid Dimensions**: 100 subregions in 10×10 arrangement
- **Pixel Resolution**: 10 meters
- **Grid Size**: 1098 × 1097 pixels per grid

## Key Concepts

### Full Grid Design

This loader provides **full grids** to give  maximum flexibility:

- **Advantages**: Preserves spatial context, allows custom patching strategies
- **Required Task**: Implement patching suitable for your model architecture

### Train/Val/Test Splits

Splits should be created for use cases:

```python
# Option 1: Sequential (by grid position)
splits = split_manager.grouped_split(train_end=70, val_end=85)

# Option 2: Random (shuffled)
splits = split_manager.random_split()

# Option 3: Custom (by grid coverage or other criteria)
# Analyze water coverage and select grids accordingly
```

### Label Handling

Labels are provided in their **raw form** (0-100 percentage):

- For **regression**: Use raw values directly
- For **classification**: Binarize at appropriate threshold (typically 25%)
- For **analysis**: Examine distribution to understand data imbalance


## Citation

If you use this dataset, please cite:

- Levenson, E., S.W. Cooley, and A. Mullen. 2025. ABoVE: Alaska Lake and Pond Occurrence. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2399

## References

- ALPOD Dataset: https://daac.ornl.gov/
- Sentinel-2: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
- Original ALPOD Paper: Levenson et al., 2025, GRL