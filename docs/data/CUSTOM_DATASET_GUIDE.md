# Creating a Custom DDLP Dataset - Complete Guide

## Overview

This guide shows how to create and use a DDLP-compatible HDF5 dataset for your two-body system physics simulations.

## Important: Understanding Dataset Return Values

**CRITICAL:** The Balls dataset (and your two-body dataset) returns 5 values: `img, pos, size, id, in_camera`

### What Each Variable Means:

1. **`img`** (actually the video): Shape `(T, C, H, W)` - The video frames tensor
2. **`pos`**: Shape `(T, n_objects, 2)` - Ground-truth XY positions in **pixel coordinates**
3. **`size`**: Shape `(T, n_objects)` - Ground-truth sizes/radii in pixels
4. **`id`**: Shape `(T, n_objects)` - Object identity labels (1.0, 2.0 for two balls)
5. **`in_camera`**: Shape `(T, n_objects)` - Visibility flags (all 1.0 for always visible)

### When Are These Used?

**During Training:** Only `img` (video) is used! DDLP is **unsupervised** - no ground-truth is needed.

**During Evaluation:** The metadata (`pos`, `size`, `id`, `in_camera`) is used for:
- **`pos`**: Measuring correlation between learned z_p and true positions (your main goal!)
- **`size`**: Evaluating scale prediction accuracy
- **`id`**: Computing tracking consistency/identity preservation
- **`in_camera`**: Handling occlusions and out-of-frame objects

### For Your 2-Body System:

Your dataset has:
- 2 balls with same size but different colors
- Moving in 2D space
- Never leaving the frame (all in_camera = 1.0)
- Ground-truth positions extracted from physics simulations

---

## Dataset Creation

**Step 0: Generate HDF5 Dataset**

Use the provided script to convert your physics trajectories to DDLP-compatible HDF5 format:

```bash
cd /data2/users/lr4617/data_scripts/video_vae
python create_video_dataset_2body_system_LATEST_ddlp.py
```

This creates an HDF5 dataset matching the Balls-Interaction structure:

**Output Dataset Structure:**
```
/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square/
├── train.hdf5  (1500 episodes)
├── val.hdf5    (100 episodes)
└── test.hdf5   (100 episodes, copy of val)
```

**HDF5 File Contents:**
Each HDF5 file contains:
- `imgs`: (n_episodes, T, 64, 64, 3) uint8 - Video frames
- `positions`: (n_episodes, T, 2, 2) float64 - Pixel coordinates [episode, time, ball, xy]
- `sizes`: (n_episodes, T, 2) float64 - Constant radius in pixels
- `ids`: (n_episodes, T, 2) float64 - Object IDs [1.0, 2.0]
- `in_camera`: (n_episodes, T, 2) float64 - All 1.0 (always visible)

---

## Step-by-Step Implementation

### Step 1: Create Dataset Class

Create a new file: `/data2/users/lr4617/ddlp/datasets/two_body_ds.py`

```python
"""
Two-Body System Dataset Loader for DDLP
Loads from HDF5 files created by create_video_dataset_2body_system_LATEST_ddlp.py
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class TwoBodyDataset(Dataset):
    """
    Dataset for two-body system physics videos (HDF5 format)
    
    Args:
        root: Path to dataset root directory containing train.hdf5, val.hdf5, test.hdf5
        mode: 'train', 'val', or 'test'
        sample_length: Number of consecutive frames to sample
        ep_len: Episode length in the HDF5 file (determined automatically)
    """
    def __init__(self, root, mode, sample_length=20, ep_len=None):
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'valid':
            mode = 'val'
        
        self.root = root
        self.mode = mode
        self.sample_length = sample_length
        
        # Load HDF5 file
        file = os.path.join(self.root, f'{mode}.hdf5')
        assert os.path.exists(file), f'Path {file} does not exist'
        self.file = file
        
        # Determine episode length from file
        with h5py.File(self.file, 'r') as f:
            self.ep_len = f['imgs'].shape[1]  # (n_episodes, T, H, W, C)
            print(f"Loaded {len(f['imgs'])} episodes from {mode} split (ep_len={self.ep_len})")
        
        # Calculate number of subsequences per episode
        if self.mode == 'train':
            self.seq_per_episode = max(1, self.ep_len - self.sample_length + 1)
        else:
            self.seq_per_episode = 1  # Use full episode for val/test

    def __getitem__(self, index):
        with h5py.File(self.file, 'r') as f:
            imgs = f['imgs']
            positions = f['positions']
            sizes = f['sizes']
            ids = f['ids']
            in_camera = f['in_camera']
            
            if self.mode == 'train':
                # Continuous indexing - sample subsequences within episodes
                ep = index // self.seq_per_episode
                offset = index % self.seq_per_episode
                end = offset + self.sample_length
                
                img = imgs[ep][offset:end]
                pos = positions[ep][offset:end]
                size = sizes[ep][offset:end]
                obj_id = ids[ep][offset:end]
                in_cam = in_camera[ep][offset:end]
            else:
                # For val/test, use full episode or sample from beginning
                img = imgs[index]
                pos = positions[index]
                size = sizes[index]
                obj_id = ids[index]
                in_cam = in_camera[index]
                
                # Optionally truncate to sample_length
                if img.shape[0] > self.sample_length:
                    img = img[:self.sample_length]
                    pos = pos[:self.sample_length]
                    size = size[:self.sample_length]
                    obj_id = obj_id[:self.sample_length]
                    in_cam = in_cam[:self.sample_length]
        
        # Convert to torch tensor: (T, H, W, C) -> (T, C, H, W)
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = img.float() / 255.0
        
        return img, pos, size, obj_id, in_cam

    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            length = len(f['imgs'])
            if self.mode == 'train':
                return length * self.seq_per_episode
            else:
                return length


class TwoBodyDatasetImage(Dataset):
    """
    Single-frame version of TwoBodyDataset for image-based training (DLP)
    """
    def __init__(self, root, mode, sample_length=1, ep_len=None):
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'valid':
            mode = 'val'
        
        self.root = root
        self.mode = mode
        self.sample_length = sample_length
        
        # Load HDF5 file
        file = os.path.join(self.root, f'{mode}.hdf5')
        assert os.path.exists(file), f'Path {file} does not exist'
        self.file = file
        
        # Determine episode length
        with h5py.File(self.file, 'r') as f:
            self.ep_len = f['imgs'].shape[1]
        
        # For single images, sample from all frames
        self.seq_per_episode = self.ep_len - self.sample_length + 1

    def __getitem__(self, index):
        with h5py.File(self.file, 'r') as f:
            imgs = f['imgs']
            positions = f['positions']
            sizes = f['sizes']
            ids = f['ids']
            in_camera = f['in_camera']
            
            # Continuous indexing across all frames
            ep = index // self.seq_per_episode
            offset = index % self.seq_per_episode
            end = offset + self.sample_length
            
            img = imgs[ep][offset:end]
            pos = positions[ep][offset:end]
            size = sizes[ep][offset:end]
            obj_id = ids[ep][offset:end]
            in_cam = in_camera[ep][offset:end]
        
        # Convert to torch: (T, H, W, C) -> (T, C, H, W)
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = img.float() / 255.0
        
        return img, pos, size, obj_id, in_cam

    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            length = len(f['imgs'])
            return length * self.seq_per_episode


# Test the dataset
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    root = '/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square'
    
    # Test video dataset
    print("Testing TwoBodyDataset...")
    ds = TwoBodyDataset(root, mode='train', sample_length=10)
    print(f"Dataset length: {len(ds)}")
    
    video, pos, size, obj_id, in_cam = ds[0]
    print(f"Video shape: {video.shape}")  # Should be (10, 3, 64, 64)
    print(f"Positions shape: {pos.shape}")  # Should be (10, 2, 2)
    print(f"Value range: [{video.min():.3f}, {video.max():.3f}]")
    print(f"Position range: [{pos.min():.3f}, {pos.max():.3f}]")
    
    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < video.shape[0]:
            # Convert from (C, H, W) to (H, W, C) for display
            img = video[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f'Frame {i}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('two_body_sample.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to two_body_sample.png")
```

**Key Implementation Details:**

1. **HDF5 Format:** Loads directly from HDF5 files (train.hdf5, val.hdf5, test.hdf5)
2. **Ground-Truth Included:** Real positions, sizes, and IDs are already in the HDF5 files
3. **Continu
└── test_positions/
```

---

### Step 3: Register Dataset in get_dataset.py

Edit `/data2/users/lr4617/ddlp/datasets/get_dataset.py`:

**Add import at the top:**
```python
from datasets.two_body_ds import TwoBodyDataset, TwoBodyDatasetImage
```

**Add to `get_video_dataset()` function:**
```python
def get_video_dataset(ds, root, seq_len=1, mode='train', image_size=128):
    # load data
    if ds == "traffic":
        dataset = TrafficDataset(path_to_npy=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == 'clevrer':
        dataset = CLEVREREpDataset(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'balls':
        dataset = Balls(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d':
        dataset = Obj3D(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d128':
        image_size = 128
        dataset = Obj3D(root=root, mode=mode, sample_length=seq_len, res=image_size)
    elif ds == 'phyre':
        dataset = PhyreDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'twobody':  # ADD THIS
        dataset = TwoBodyDataset(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'langtable':
        if not LANGTABLE_AVAILABLE:
            raise ImportError("Language Table dataset is not available. Please install or remove from config.")
        dataset = LanguageTableDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    else:
        raise NotImplementedError
    return dataset
```

**Add to `get_image_dataset()` function:**
```python
def get_image_dataset(ds, root, mode='train', image_size=128, seq_len=1):
    # set seq_len > 1 when training with use_tracking
    # load data
    if ds == "traffic":
        dataset = TrafficDatasetImage(path_to_npy=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == 'clevrer':
        dataset = CLEVREREpDatasetImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'balls':
        dataset = BallsImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d':
        dataset = Obj3DImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d128':
        image_size = 128
        dataset = Obj3DImage(root=root, mode=mode, sample_length=seq_len, res=image_size)
    elif ds == 'phyre':
        dataset = PhyreDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'twobody':  # ADD THIS
        dataset = TwoBodyDatasetImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'shapes':
        if mode == 'train':
            dataset = generate_shape_dataset_torch(img_size=image_size, num_images=40_000)
        else:
            dataset = generate_shape_dataset_torch(img_size=image_size, num_images=2_000)
    elif ds == 'langtable':
        if not LANGTABLE_AVAILABLE:
            raise ImportError("Language Table dataset is not available. Please install or remove from config.")
        dataset = LanguageTableDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    else:
        raise NotImplementedError
    return dataset
```

---

### Step 4: Create Configuration File

Create `/data2/users/lr4617/ddlp/configs/twobody.json`:

```json
{
  "ds": "twobody",
  "root": "/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square/two_body_system_extrapolation_square",
  "device": "cuda",
  "batch_size": 32,
  "lr": 0.0002,
  "kp_activation": "tanh",
  "pad_mode": "replicate",
  "load_model": false,
  "pretrained_path": null,
  "num_epochs": 150,
  "n_kp": 1,
  "recon_loss_type": "mse",
  "sigma": 1.0,
  "beta_kl": 0.1,
  "beta_rec": 1.0,
  "patch_size": 8,
  "topk": 6,
  "n_kp_enc": 2,
  "eval_epoch_freq": 1,
  "learned_feature_dim": 3,
  "n_kp_prior": 12,
  "weight_decay": 0.0,
  "kp_range": [-1, 1],
  "warmup_epoch": 1,
  "dropout": 0.0,
  "iou_thresh": 0.2,
  "anchor_s": 0.25,
  "kl_balance": 0.001,
  "image_size": 64,
  "ch": 3,
  "enc_channels": [32, 64, 128],
  "prior_channels": [16, 32, 64],
  "timestep_horizon": 10,
  "predict_delta": true,
  "beta_dyn": 0.1,
  "scale_std": 0.3,
  "offset_std": 0.2,
  "obj_on_alpha": 0.1,
  "obj_on_beta": 0.1,
  "beta_dyn_rec": 1.0,
  "num_static_frames": 4,
  "pint_layers": 6,
  "pint_heads": 8,
  "pint_dim": 256,
  "run_prefix": "twobody_",
  "animation_horizon": 100,
  "eval_im_metrics": true,
  "use_resblock": false,
  "scheduler_gamma": 0.95,
  "adam_betas": [0.9, 0.999],
  "adam_eps": 0.0001,
  "train_enc_prior": true,
  "start_dyn_epoch": 0,
  "cond_steps": 10,
  "animation_fps": 0.06,
  "use_correlation_heatmaps": true,
  "enable_enc_attn": false,
  "filtering_heuristic": "variance"
}
```

**Key Configuration Parameters to Adjust:**

- **`n_kp_enc: 2`** - Number of particles (2 bodies in your system)
- **`image_size: 64`** - Matches your 64x64 resolution
- **`timestep_horizon: 10`** - **IMPORTANT:** Should match `sample_length` in your dataset! This is the sequence length for dynamics training
- **`learned_feature_dim: 3`** - Visual feature dimension (can tune)
- **`batch_size: 32`** - Adjust based on GPU memory

**Important Note on `timestep_horizon`:**
The `timestep_horizon` parameter in the config should match the `sample_length` used when loading your dataset. For example:
- If your dataset loads 10-frame sequences (`sample_length=10`), set `"timestep_horizon": 10`
- If your dataset loads 20-frame sequences (`sample_length=20`), set `"timestep_horizon": 20`

This ensures the model trains on the full sequence you're providing.

Create `/data2/users/lr4617/ddlp/configs/twobody.json`:

```json4: Test Dataset Loading

Before training, verify the dataset loads correctly:

```bash
cd /data2/users/lr4617/ddlp

# Test the dataset class
python datasets/two_body_ds.py

# Or test with Python
python -c "
from datasets.two_body_ds import TwoBodyDataset
ds = TwoBodyDataset('/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square', 'train', sample_length=10)
print(f'Dataset size: {len(ds)}')
video, pos, size, obj_id, in_cam = ds[0]
print(f'Video shape: {video.shape}')
print(f'Positions shape: {pos.shape}')
print(f'Value range: [{video.min():.3f}, {video.max():.3f}]')
print(f'Position range: [{pos.min():.3f}, {pos.max():.3f}]')
print('✓ Dataset loading works!')
"
```

Expected output:
```
Loaded 1500 episodes from train split (ep_len=30)
Dataset size: 1500
Video shape: torch.Size([10, 3, 64, 64])
Positions shape: (10, 2, 2)
Value range: [0.149, 1.000]
Position range: [5.234, 58.765]
✓ Dataset loading works!
```

**Important Notes:**
- Position values are in pixel coordinates (not normalized), matching the Balls dataset format
- Dataset size = 1500 (one sample per episode, using first `sample_length` frames)
- Make sure `timestep_horizon` in your config matches the `sample_length` you're using (both should be 10 in this example)

---

### Step 6: Train the Model

Now you can train DDLP on your custom dataset:

```bash
cd /data2/users/lr4617/ddlp

# Activate your environment
conda activate ddlp

# Start training
python train_ddlp.py -d twobody

# Or with explicit config
python train_ddlp.py --config configs/twobody.json
```

**Monitor6 (Optional): Adjust Dataset Generation Parameters

If you need to regenerate the dataset with different parameters, edit the main section of `create_video_dataset_2body_system_LATEST_ddlp.py`:

```python
# Adjust these parameters as needed:
max_training_samples=1500,      # Number of training episodes
max_validation_samples=100,     # Number of validation episodes
radius_frac=0.10,               # Max ball radius (fraction of image size)
two_colors=True,                # Use different colors for the two balls
```

Then re-run the script to regenerate the HDF5 files.
# Move some training videos to val (e.g., 10% of training data)
# Option 1: Random selection
ls train/ | shuf -n 152 | xargs -I {} mv train/{} val/

# Option 2: Last 152 files
ls train/ | tail -152 | xargs -I {} mv train/{} val/
```

---

## Advanced Customizations

### Adding Ground-Truth Positions

If you have ground-truth object positions, modify the dataset to return them:

```python
class TwoBodyDataset(Dataset):
    def __getitem__(self, index):
        # ... existing code ...
        video = video.float() / 255.0
        
        # Load positions if available
        pos_file = self.video_files[ep].replace('video_', 'positions_').replace('.npy', '.npy')
        if os.path.exists(pos_file):
            positions = np.load(pos_file)  # Shape: (T, num_objects, 2)
            positions = torch.from_numpy(positions[offset:end]).float()
            return video, positions
        
        return video
```

### Different Resolution

If you want to train at different resolution:

```python
class TwoBodyDataset(Dataset):
    def __init__(self, root, mode, sample_length=20, ep_len=60, target_size=None):
        # ... existing code ...
        self.target_size = target_size
    
    def __getitem__(self, index):
        # ... existing code ...
        video = video.float() / 255.0
        
        # Resize if needed
        if self.target_size is not None and self.target_size != video.shape[-1]:
            import torch.nn.functional as F
            video = F.interpolate(video, size=(self.target_size, self.target_size),
                                mode='bilinear', align_corners=False)
        
        return video
```

Then update config: `"image_size": 128`

---

## Troubleshooting

### Issue: Dataset loading is slow
**Solution:** Consider converting to HDF5 format (like Balls dataset) for faster loading:

```python
import h5py

# Convert numpy files to HDF5
with h5py.File('train.hdf5', 'w') as f:
    videos = []
    for video_file in video_files:
        videos.append(np.load(video_file))
    videos = np.stack(videos)
    f.create_dataset('imgs', data=videos, compression='gzip')
```

### Issue: Out of memory during training
**Solution:** Reduce batch size or sequence length:
- `"batch_size": 16` (or 8)
- `"timestep_horizon": 5`

### Issue: Model doesn't track objects well
**Solution:** Adjust hyperparameters:
- Increase `n_kp_enc` if there are more objects
- Adjust `anchor_s` (glimpse size)
- Try `"use_correlation_heatmaps": true` for better tracking

### Issue: Poor reconstruction quality
**Solution:**
- Increase training epochs
- Adjust KL balance: `"beta_kl": 0.05`
- Try VGG loss: `"recon_loss_type": "vgg"`

---

## Expected Training Time

With your dataset (1520 training videos):
- **Per epoch:** ~5-10 minutes (depends on GPU)
- **Total (150 epochs):** ~12-25 hours
- **Checkpoints:** Saved periodically (best model, last model)
00 training episodes, ~31,500 subsequences):
- **Per epoch:** ~10-20 minutes (depends on GPU and episode length)
- **Total (150 epochs):** ~25-50 hours
- **Checkpoints:** Saved periodically (best model, last model)

**Note:** Training time depends on episode length in your HDF5 files. Longer episodes create more training subsequences.

1. ✅ Create `datasets/two_body_ds.py`
2. ✅ Update `datasets/get_dataset.py`
3. ✅ Generate HDF5 dataset using `create_video_dataset_2body_system_LATEST_ddlp.py`
2. ✅ Create `datasets/two_body_ds.py`
3. ✅ Update `datasets/get_dataset.py`
4. ✅ Create `configs/twobody.json`
5. ✅ Test dataset loading
6. ✅ Start training
7. Monitor results and adjust hyperparameters
8. Evaluate z_p correlation with ground-truth positions (using real positions from HDF5!)
9--

## Quick Reference Commands

```bash
# Test dataset
pyGenerate dataset (first time only)
cd /data2/users/lr4617/data_scripts/video_vae
python create_video_dataset_2body_system_LATEST_ddlp.py

# Test dataset
cd /data2/users/lr4617/ddlp
python datasets/two_body_ds.py

# Train model
python train_ddlp.py -d twobody

# Resume training
python train_ddlp.py -d twobody --load_model --pretrained_path ./path/to/checkpoint.pth

# Evaluate
python eval/eval_model.py -d twobody --checkpoint ./path/to/checkpoint.pth

# Generate predictions
python generate_ddlp_video_prediction.py -d twobody --checkpoint ./path/to/checkpoint.pth

# Inspect HDF5 dataset
python -c "
import h5py
with h5py.File('/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square/train.hdf5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('Shapes:')
    for key in f.keys():
        print(f'  {key}: {f[key].shape}')
"

---

**Good luck with your training! 🚀**
