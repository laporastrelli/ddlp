# DDLP Dataset Metadata - Complete Explanation

## Your Questions Answered

### 1. Is "img" actually the video?

**YES!** Despite the name `img`, it's actually the full video sequence:
- Shape: `(T, C, H, W)` where T = number of frames, C = channels, H/W = height/width
- Example: `(10, 3, 64, 64)` means 10 frames of 64x64 RGB video
- It's called "img" probably because the code also handles single images in image-only mode

### 2. What do pos, size, id, and in_camera indicate?

Let me explain each with the **Balls-Interactions dataset** as an example (6 balls moving in 64x64 space):

#### `pos` - Ground-Truth Positions
- **Shape:** `(T, n_objects, 2)` - e.g., `(10, 6, 2)` for 10 frames, 6 balls
- **Content:** XY coordinates of each object at each timestep
- **Values:** Typically normalized to [-1, 1] range (matching `kp_range` in config)
- **Example:** `pos[5, 2, :]` = `[0.3, -0.5]` means ball #2 at frame 5 is at position (0.3, -0.5)

#### `size` - Ground-Truth Object Sizes/Scales
- **Shape:** `(T, n_objects)` - e.g., `(10, 6)`
- **Content:** Radius or scale factor of each object at each timestep
- **Values:** Typically in range [0, 1] representing fraction of image size
- **Example:** `size[5, 2]` = `0.1` means ball #2 at frame 5 has radius ~10% of image width

#### `id` - Object Identity Labels
- **Shape:** `(T, n_objects)` - e.g., `(10, 6)`
- **Content:** Unique identifier for each object (which ball is which)
- **Values:** Integers 0, 1, 2, 3, 4, 5 for 6 balls
- **Purpose:** Track which object is which across time (important when objects cross paths)
- **Example:** `id[5, :]` = `[0, 1, 2, 3, 4, 5]` means the same 6 balls are present

#### `in_camera` - Visibility Flags
- **Shape:** `(T, n_objects)` - e.g., `(10, 6)` 
- **Content:** Boolean indicating if object is visible in the frame
- **Values:** `True` if object is in frame, `False` if out of bounds or occluded
- **Example:** `in_camera[5, :]` = `[True, True, False, True, True, True]` means ball #2 left the frame

---

## 3. When Are These Variables Used?

### During Training: **ONLY `img` is used!**

Looking at [train_ddlp.py:198](train_ddlp.py#L198):
```python
for batch in dataloader:
    x = batch[0].to(device)  # Only img is used!
    # pos, size, id, in_camera are IGNORED during training
```

**Why?** DDLP is **unsupervised** - it learns object representations without any labels.

### During Evaluation: **All metadata is used**

The metadata is used for:

1. **Measuring Correlation** (your main goal!):
   - Compare learned `z_p` positions with ground-truth `pos`
   - Calculate Pearson correlation coefficient
   - This tells you if DDLP learned meaningful spatial representations

2. **Tracking Accuracy**:
   - Use `id` to check if DDLP maintains object identity across time
   - Important when objects cross paths or occlude each other

3. **Scale Prediction**:
   - Compare learned `z_s` with ground-truth `size`
   - Evaluate if model correctly estimates object sizes

4. **Occlusion Handling**:
   - Use `in_camera` to exclude out-of-frame objects from metrics
   - Only evaluate on visible objects

---

## 4. How to Integrate These for Your 2-Body System

Your dataset: **2 balls, same size, different colors, 2D motion, never leave frame**

### Option A: Simple Dummy Metadata (Quick Start)

If you just want to train and don't care about evaluation yet:

```python
def __getitem__(self, index):
    # ... load video ...
    video = video.float() / 255.0
    
    # Dummy metadata (not used during training anyway)
    n_frames = video.shape[0]
    pos = np.zeros((n_frames, 2, 2), dtype=np.float32)  # Will be ignored
    size = np.ones((n_frames, 2), dtype=np.float32) * 0.1
    obj_id = np.array([[0, 1]] * n_frames)
    in_camera = np.ones((n_frames, 2), dtype=bool)
    
    return video, pos, size, obj_id, in_camera
```

**Pros:** Quick to implement, training works fine  
**Cons:** Can't measure z_p correlation after training

---

### Option B: Real Metadata (For Evaluation)

If you have or can extract ground-truth positions:

```python
class TwoBodyDataset(Dataset):
    def __init__(self, root, mode, sample_length=20, ep_len=60):
        # ... existing code ...
        
        # Check for ground-truth positions
        self.positions_dir = os.path.join(root, mode + '_positions')
        self.has_gt = os.path.exists(self.positions_dir)
        
        if self.has_gt:
            self.position_files = sorted(glob.glob(
                os.path.join(self.positions_dir, 'positions_*.npy')
            ))
            print(f"✓ Found {len(self.position_files)} ground-truth position files")

    def __getitem__(self, index):
        # ... load video ...
        
        if self.has_gt:
            # Load real positions
            positions = np.load(self.position_files[ep_idx])  # Shape: (60, 2, 2)
            positions = positions[offset:end]  # Sample same subsequence as video
            
            # Normalize to [-1, 1] (assuming positions are in pixel coords [0, 64])
            positions = (positions / 32.0) - 1.0
            
            # Size: constant for both balls (assume radius = 5 pixels ~ 0.078 of image)
            size = np.ones((positions.shape[0], 2), dtype=np.float32) * 0.078
            
            # IDs: ball 0 and ball 1
            obj_id = np.tile(np.array([0, 1]), (positions.shape[0], 1))
            
            # In camera: always True (balls never leave frame)
            in_camera = np.ones((positions.shape[0], 2), dtype=bool)
        else:
            # Fallback to dummy metadata
            n_frames = video.shape[0]
            positions = np.zeros((n_frames, 2, 2), dtype=np.float32)
            size = np.ones((n_frames, 2), dtype=np.float32) * 0.078
            obj_id = np.tile(np.array([0, 1]), (n_frames, 1))
            in_camera = np.ones((n_frames, 2), dtype=bool)
        
        return video, positions, size, obj_id, in_camera
```

**Directory Structure:**
```
two_body_system_extrapolation_square/
├── train/
│   ├── video_0.npy          # (60, 64, 64, 3)
│   ├── video_1.npy
│   └── ...
├── train_positions/
│   ├── positions_0.npy      # (60, 2, 2) - [time, ball, xy]
│   ├── positions_1.npy
│   └── ...
├── test/
│   └── video_*.npy
└── test_positions/
    └── positions_*.npy
```

---

## 5. How to Extract Positions (If You Don't Have Them)

Since your balls have different colors, you can extract positions automatically:

```python
import numpy as np
import cv2
from scipy.ndimage import center_of_mass

def extract_positions_from_video(video_path):
    """
    Extract ball positions from video using color segmentation
    
    Args:
        video_path: Path to video .npy file
    
    Returns:
        positions: (T, 2, 2) array of [frame, ball, xy] positions in pixels
    """
    video = np.load(video_path)  # (60, 64, 64, 3)
    T = video.shape[0]
    positions = np.zeros((T, 2, 2), dtype=np.float32)
    
    for t in range(T):
        frame = video[t]  # (64, 64, 3)
        
        # Convert to HSV for better color separation
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Method 1: Threshold by color (adjust ranges for your ball colors!)
        # Example: detecting red and blue balls
        mask_ball1 = cv2.inRange(frame_hsv, 
                                  np.array([0, 100, 100]),    # lower red
                                  np.array([10, 255, 255]))   # upper red
        mask_ball2 = cv2.inRange(frame_hsv,
                                  np.array([100, 100, 100]),  # lower blue
                                  np.array([130, 255, 255]))  # upper blue
        
        # Find centroids
        for ball_idx, mask in enumerate([mask_ball1, mask_ball2]):
            y, x = center_of_mass(mask)
            if not np.isnan(x) and not np.isnan(y):
                positions[t, ball_idx] = [x, y]
    
    return positions

# Process all videos
import glob
from tqdm import tqdm

video_dir = '/data2/users/lr4617/data/video_vae/two_body_system_extrapolation_square/train'
output_dir = video_dir + '_positions'
os.makedirs(output_dir, exist_ok=True)

for video_file in tqdm(sorted(glob.glob(f'{video_dir}/video_*.npy'))):
    video_idx = video_file.split('_')[-1].split('.')[0]
    positions = extract_positions_from_video(video_file)
    np.save(f'{output_dir}/positions_{video_idx}.npy', positions)
```

**Note:** Adjust the HSV color ranges based on your actual ball colors. You can inspect a sample frame to determine the right ranges.

---

## 6. Summary Table

| Variable | Shape | Content | Used in Training? | Used in Evaluation? | For Your Dataset |
|----------|-------|---------|-------------------|---------------------|------------------|
| `img` (video) | `(T, C, H, W)` | RGB video frames | ✅ YES | ✅ YES | Load from `.npy` files |
| `pos` | `(T, n_obj, 2)` | XY positions | ❌ NO | ✅ YES (correlation) | Extract or set to zero |
| `size` | `(T, n_obj)` | Object radii/scales | ❌ NO | ✅ YES (optional) | Constant 0.078 |
| `id` | `(T, n_obj)` | Object identities | ❌ NO | ✅ YES (tracking) | `[0, 1]` for 2 balls |
| `in_camera` | `(T, n_obj)` | Visibility flags | ❌ NO | ✅ YES (filtering) | All `True` |

---

## 7. Recommendation

**For your workflow:**

1. **Phase 1 (Training):** Use dummy metadata - it doesn't affect training at all
2. **Phase 2 (After training):** Extract ground-truth positions for evaluation
3. **Phase 3 (Measure correlation):** Re-run evaluation with real positions to compute z_p correlation

This way you can start training immediately without worrying about positions!

---

## 8. Quick Test

Verify your dataset returns the right structure:

```python
from datasets.two_body_ds import TwoBodyDataset

ds = TwoBodyDataset('/path/to/your/dataset', 'train', sample_length=10)
video, pos, size, obj_id, in_camera = ds[0]

print(f"video shape: {video.shape}")        # Should be (10, 3, 64, 64)
print(f"pos shape: {pos.shape}")            # Should be (10, 2, 2)
print(f"size shape: {size.shape}")          # Should be (10, 2)
print(f"obj_id shape: {obj_id.shape}")      # Should be (10, 2)
print(f"in_camera shape: {in_camera.shape}")  # Should be (10, 2)
```

All shapes should match the Balls dataset structure!
