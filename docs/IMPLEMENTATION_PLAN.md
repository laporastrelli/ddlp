# DDLP Implementation Plan - Balls-Interaction Dataset

## Overview
Evaluate and compare different latent representations for DDLP on Balls-Interaction dataset, then apply the best-performing version to custom dataset.

---

## Phase 1: Baseline Evaluation (Overparametrized - 9D)
**Goal:** Establish baseline performance with the original full representation

### 1.1 Setup & Preparation
- [x] Download Balls-Interaction dataset from [MEGA.nz](https://mega.nz/file/4cUR1b5a#RwFFzCiESeeQb8rYgt7PK2_D8b_69-K85RV3jlaphTo)
- [x] Verify dataset structure (train.hdf5, val.hdf5, test.hdf5)
- [x] Update `root` path in `configs/balls.json` to point to downloaded data
- [x] (*NOT THERE*) Check if pre-trained checkpoints are available for Balls-Interaction

### 1.2 Training (if no checkpoints available)
- [x] Run training: `python train_ddlp.py -d balls`
- [ ] Monitor training progress (reconstruction quality, loss curves)
- [ ] Save checkpoints at appropriate intervals
- [ ] Training completes successfully

### 1.3 Evaluation & Analysis
- [ ] Load trained/pre-trained model
- [ ] Extract encoded z_p (position latents) from test set
- [ ] Load ground-truth positions from dataset (stored in HDF5 files)
- [ ] Compute correlation between z_p and ground-truth positions:
  - [ ] Pearson correlation coefficient
  - [ ] Spearman correlation coefficient
  - [ ] Mean Squared Error (MSE)
  - [ ] R² score
- [ ] Visualize alignment (scatter plots, trajectory comparisons)
- [ ] Document baseline metrics

**Baseline Results (9D - Full Representation):**
```
Pearson correlation (x): ____
Pearson correlation (y): ____
Spearman correlation (x): ____
Spearman correlation (y): ____
MSE: ____
R² score: ____
```

---

## Phase 2: Reduced Representations Training & Evaluation

### 2.1 Code Modifications (One-time setup)
**Follow guide:** `docs/minimal_intermediate_representations.md`

#### 2.1.1 Intermediate Representation (7D) - Removes z_d, z_t
- [ ] Modify `modules/modules.py` - ParticleAttributeEncoder
  - [ ] Add `use_depth`, `use_transparency`, `use_scale` parameters
  - [ ] Conditionally create neural network heads
  - [ ] Update `forward()` method with conditional logic
- [ ] Modify `modules/dynamics_modules.py` - ParticleFeatureProjection
  - [ ] Update `particle_dim` calculation
  - [ ] Add conditional projections
  - [ ] Update `forward()` concatenation logic
- [ ] Modify `modules/dynamics_modules.py` - ParticleFeatureDecoder
  - [ ] Add conditional heads
  - [ ] Update `forward()` with default values for disabled dimensions
- [ ] Modify `modules/dynamics_modules.py` - DynamicsDLP
  - [ ] Pass dimension flags to sub-modules
- [ ] Modify `models.py` - Loss computations
  - [ ] Add conditional KL loss terms for scale, depth, transparency
  - [ ] Set disabled losses to 0.0
- [ ] Modify training scripts (`train_ddlp.py`, `train_dlp.py`)
  - [ ] Read dimension flags from config
  - [ ] Pass flags to model initialization
- [ ] Run unit tests - verify intermediate model initializes
- [ ] Run forward pass test - verify output shapes are correct

#### 2.1.2 Minimal Representation (5D) - Removes z_d, z_t, z_s
- [ ] Same modifications as above (already done if intermediate works)
- [ ] Ensure fixed scale value is used when `use_scale=False`
- [ ] Run unit tests - verify minimal model initializes
- [ ] Run forward pass test - verify output shapes are correct

### 2.2 Train Intermediate Representation (7D)
- [ ] Verify config: `configs/balls_intermediate.json`
- [ ] Update dataset `root` path if needed
- [ ] Run training: `python train_ddlp.py -d balls --config configs/balls_intermediate.json`
- [ ] Monitor training progress
- [ ] Training completes successfully
- [ ] Save checkpoint

### 2.3 Evaluate Intermediate Representation
- [ ] Load trained intermediate model
- [ ] Extract encoded z_p from test set
- [ ] Compute correlation metrics with ground-truth positions
- [ ] Visualize alignment
- [ ] Compare with baseline results

**Intermediate Results (7D):**
```
Pearson correlation (x): ____
Pearson correlation (y): ____
Spearman correlation (x): ____
Spearman correlation (y): ____
MSE: ____
R² score: ____
Improvement over baseline: ____
```

### 2.4 Train Minimal Representation (5D)
- [ ] Verify config: `configs/balls_minimal.json`
- [ ] Update dataset `root` path if needed
- [ ] Run training: `python train_ddlp.py -d balls --config configs/balls_minimal.json`
- [ ] Monitor training progress
- [ ] Training completes successfully
- [ ] Save checkpoint

### 2.5 Evaluate Minimal Representation
- [ ] Load trained minimal model
- [ ] Extract encoded z_p from test set
- [ ] Compute correlation metrics with ground-truth positions
- [ ] Visualize alignment
- [ ] Compare with baseline and intermediate results

**Minimal Results (5D):**
```
Pearson correlation (x): ____
Pearson correlation (y): ____
Spearman correlation (x): ____
Spearman correlation (y): ____
MSE: ____
R² score: ____
Improvement over baseline: ____
```

### 2.6 Comparative Analysis
- [ ] Create comparison table of all three representations
- [ ] Plot correlation metrics side-by-side
- [ ] Analyze training efficiency (time per epoch, memory usage)
- [ ] Analyze parameter counts
- [ ] Document qualitative differences (if any)
- [ ] **Select best-performing representation** for Phase 3

**Best Representation:** [ ] 9D (Full) | [ ] 7D (Intermediate) | [ ] 5D (Minimal)

---

## Phase 3: Custom Dataset Training

### 3.1 Dataset Preparation
- [ ] Prepare custom dataset in appropriate format
- [ ] Ensure dataset has ground-truth positions (for evaluation)
- [ ] Create HDF5 files or adapt dataset loader
- [ ] Split into train/val/test sets
- [ ] Verify dataset statistics (number of objects, image size, etc.)

### 3.2 Configuration
- [ ] Create new config file for custom dataset
  - [ ] Copy config from best-performing representation
  - [ ] Update dataset-specific parameters:
    - [ ] `ds`: custom dataset name
    - [ ] `root`: path to custom dataset
    - [ ] `image_size`: match custom dataset
    - [ ] `n_kp_enc`: match number of objects (if known)
    - [ ] `learned_feature_dim`: adjust if needed
- [ ] Verify config is correct

### 3.3 Dataset Loader (if needed)
- [ ] Create custom dataset class in `datasets/` folder
- [ ] Implement `__init__`, `__getitem__`, `__len__`
- [ ] Add to `datasets/get_dataset.py`
- [ ] Test dataset loading

### 3.4 Training on Custom Dataset
- [ ] Run training with best-performing representation
- [ ] Monitor training progress
- [ ] Adjust hyperparameters if needed:
  - [ ] Learning rate
  - [ ] KL balance weights
  - [ ] Number of epochs
- [ ] Training completes successfully
- [ ] Save final checkpoint

### 3.5 Evaluation on Custom Dataset
- [ ] Load trained model
- [ ] Extract encoded z_p
- [ ] Compare with ground-truth positions (if available)
- [ ] Compute correlation metrics
- [ ] Evaluate video prediction quality:
  - [ ] Visual inspection
  - [ ] Quantitative metrics (PSNR, SSIM, FVD if applicable)
- [ ] Document final results

**Custom Dataset Results:**
```
Representation used: ____
Pearson correlation (x): ____
Pearson correlation (y): ____
Spearman correlation (x): ____
Spearman correlation (y): ____
MSE: ____
R² score: ____
Video prediction quality: ____
```

---

## Additional Notes & Observations

### Training Tips
- Start with fewer epochs (10-20) for initial experiments to verify everything works
- Use smaller batch size if GPU memory is limited
- Monitor reconstruction quality visually during training
- Check that particles track objects consistently across frames

### Evaluation Script
Consider creating a dedicated evaluation script: `eval_position_correlation.py`
```python
# Pseudocode structure:
# 1. Load model checkpoint
# 2. Load test dataset
# 3. For each sample:
#    - Get encoded z_p from model
#    - Get ground-truth positions from dataset
#    - Match particles to objects (Hungarian algorithm)
#    - Compute correlation
# 4. Aggregate and report results
```

### Expected Outcomes
- **If 9D performs best:** Complex dimensions help even for simple datasets
- **If 7D performs best:** Depth and transparency not needed for 2D
- **If 5D performs best:** Minimal representation sufficient for constrained datasets

### Troubleshooting
- If correlations are low for all representations:
  - Check particle-object assignment (may need Hungarian matching)
  - Verify ground-truth position format/scale matches model output
  - Increase training epochs
  - Adjust KL balance weights

---

## Timeline Estimates

| Phase | Estimated Time | Status |
|-------|---------------|--------|
| Phase 1: Baseline (if training needed) | 1-3 days | ⬜ Not Started |
| Phase 1: Baseline (if using checkpoint) | 2-4 hours | ⬜ Not Started |
| Phase 2.1: Code modifications | 1-2 days | ⬜ Not Started |
| Phase 2.2-2.5: Train & evaluate both | 2-4 days | ⬜ Not Started |
| Phase 2.6: Comparative analysis | 2-4 hours | ⬜ Not Started |
| Phase 3: Custom dataset | 3-7 days | ⬜ Not Started |
| **Total** | **1-3 weeks** | |

---

## Resources & References

- [Balls-Interaction Dataset](https://mega.nz/file/4cUR1b5a#RwFFzCiESeeQb8rYgt7PK2_D8b_69-K85RV3jlaphTo)
- [G-SWM Repository](https://github.com/zhixuan-lin/G-SWM) (original Balls dataset)
- [DDLP Paper](https://arxiv.org/abs/2306.05957)
- Implementation guide: `docs/minimal_intermediate_representations.md`
- Original config: `configs/balls.json`
- Intermediate config: `configs/balls_intermediate.json`
- Minimal config: `configs/balls_minimal.json`

---

**Last Updated:** January 21, 2026
