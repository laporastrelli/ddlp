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
            self.n_episodes = f['imgs'].shape[0]
            print(f"Loaded {self.n_episodes} episodes from {mode} split (ep_len={self.ep_len})")
        
        if self.mode == 'train':
            # Training: truncate to 60 frames, then create non-overlapping subsequences
            self.max_train_frames = 60
            
            # Special case: if sample_length > 60 (e.g., animation_horizon=120),
            # use full episode length for animation visualization (don't create subsequences)
            if sample_length > self.max_train_frames:
                self.max_eval_frames = min(self.ep_len, sample_length)
                self.seq_per_episode = 1
                self.use_subsequences = False
                print(f"Training (animation/eval mode): Using full episodes truncated to {self.max_eval_frames} frames (sample_length={sample_length} > max_train_frames={self.max_train_frames})")
            else:
                self.seq_per_episode = self.max_train_frames // sample_length
                self.use_subsequences = True
                print(f"Training: Using first {self.max_train_frames} frames, creating {self.seq_per_episode} subsequences of length {sample_length}")
        else:
            # Val/Test: truncate to sample_length (which will be animation_horizon when animating)
            self.max_eval_frames = min(self.ep_len, sample_length)
            self.seq_per_episode = 1
            self.use_subsequences = False
            print(f"Val/Test: Truncating episodes to {self.max_eval_frames} frames")

    def __getitem__(self, index):
        with h5py.File(self.file, 'r') as f:
            imgs = f['imgs']
            positions = f['positions']
            sizes = f['sizes']
            ids = f['ids']
            in_camera = f['in_camera']
            
            if self.mode == 'train' and self.use_subsequences:
                # Training with subsequences: Calculate which episode and which subsequence
                episode_idx = index // self.seq_per_episode
                subseq_idx = index % self.seq_per_episode
                
                # Extract non-overlapping subsequence from first 60 frames
                start_frame = subseq_idx * self.sample_length
                end_frame = start_frame + self.sample_length
                
                img = imgs[episode_idx][start_frame:end_frame]
                pos = positions[episode_idx][start_frame:end_frame]
                size = sizes[episode_idx][start_frame:end_frame]
                obj_id = ids[episode_idx][start_frame:end_frame]
                in_cam = in_camera[episode_idx][start_frame:end_frame]
            else:
                # Val/Test or Training animation mode: return first max_eval_frames (truncated)
                img = imgs[index][:self.max_eval_frames]
                pos = positions[index][:self.max_eval_frames]
                size = sizes[index][:self.max_eval_frames]
                obj_id = ids[index][:self.max_eval_frames]
                in_cam = in_camera[index][:self.max_eval_frames]
        
        # Convert to torch tensor: (T, H, W, C) -> (T, C, H, W)
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = img.float() / 255.0
        
        # Convert metadata to numpy arrays if needed
        pos = np.asarray(pos, dtype=np.float32)
        size = np.asarray(size, dtype=np.float32)
        obj_id = np.asarray(obj_id, dtype=np.float32)
        in_cam = np.asarray(in_cam, dtype=np.float32)
        
        return img, pos, size, obj_id, in_cam

    def __len__(self):
        if self.mode == 'train':
            return self.n_episodes * self.seq_per_episode
        else:
            return self.n_episodes


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
            self.n_episodes = f['imgs'].shape[0]
            print(f"Loaded {self.n_episodes} episodes from {mode} split (ep_len={self.ep_len})")
        
        # Only use first sample_length frames from each episode
        self.seq_per_episode = 1

    def __getitem__(self, index):
        with h5py.File(self.file, 'r') as f:
            imgs = f['imgs']
            positions = f['positions']
            sizes = f['sizes']
            ids = f['ids']
            in_camera = f['in_camera']
            
            # Use first sample_length frames from each episode
            img = imgs[index][:self.sample_length]
            pos = positions[index][:self.sample_length]
            size = sizes[index][:self.sample_length]
            obj_id = ids[index][:self.sample_length]
            in_cam = in_camera[index][:self.sample_length]
        
        # Convert to torch: (T, H, W, C) -> (T, C, H, W)
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = img.float() / 255.0
        
        # Convert metadata to numpy arrays
        pos = np.asarray(pos, dtype=np.float32)
        size = np.asarray(size, dtype=np.float32)
        obj_id = np.asarray(obj_id, dtype=np.float32)
        in_cam = np.asarray(in_cam, dtype=np.float32)
        
        return img, pos, size, obj_id, in_cam

    def __len__(self):
        return self.n_episodes * self.seq_per_episode


# Test the dataset
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    root = '/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square/'
    
    # Test video dataset
    print("Testing TwoBodyDataset...")
    ds = TwoBodyDataset(root, mode='train', sample_length=10)
    print(f"Dataset length: {len(ds)}")
    
    video, pos, size, obj_id, in_cam = ds[0]
    print(f"\nVideo shape: {video.shape}")  # Should be (T, 3, 64, 64)
    print(f"Positions shape: {pos.shape}")  # Should be (T, 2, 2)
    print(f"Sizes shape: {size.shape}")  # Should be (T, 2)
    print(f"IDs shape: {obj_id.shape}")  # Should be (T, 2)
    print(f"In-camera shape: {in_cam.shape}")  # Should be (T, 2)
    
    print(f"\nVideo value range: [{video.min():.3f}, {video.max():.3f}]")
    print(f"Position range: [{pos.min():.3f}, {pos.max():.3f}]")
    print(f"Size range: [{size.min():.3f}, {size.max():.3f}]")
    print(f"ID values: {np.unique(obj_id)}")
    print(f"In-camera values: {np.unique(in_cam)}")
    
    # Visualize
    print("\nVisualizing sample...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < video.shape[0]:
            # Convert from (C, H, W) to (H, W, C) for display
            img = video[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            
            # Plot positions
            for ball_idx in range(2):
                x, y = pos[i, ball_idx]
                ax.plot(x, y, 'rx', markersize=10, markeredgewidth=2)
            
            ax.set_title(f'Frame {i}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('two_body_sample.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to two_body_sample.png")
    
    # Test image dataset
    print("\n" + "="*70)
    print("Testing TwoBodyDatasetImage...")
    ds_img = TwoBodyDatasetImage(root, mode='train', sample_length=1)
    print(f"Dataset length: {len(ds_img)}")
    
    video, pos, size, obj_id, in_cam = ds_img[0]
    print(f"Video shape: {video.shape}")  # Should be (1, 3, 64, 64)
    print("✓ TwoBodyDatasetImage works!")
