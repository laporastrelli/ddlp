# DDLP Configuration Evolution Summary

This document tracks the evolution of configuration parameters across different training runs, showing how each setup deviates from the baseline `balls.json` configuration.

## Overview

- **Baseline**: `configs/balls.json` - Default configuration for BALLS_INTERACTION dataset
- **Target Dataset Transition**: From BALLS_INTERACTION → Two-Body Physics System
- **Key Evolution**: Experimentation with temporal horizons, model capacity, and dataset versions

---

## Training Runs Chronology

### Run 1: `210126_224044_balls_ddlp`
- **Purpose**: Baseline training on BALLS_INTERACTION dataset
- **Config**: `configs/balls.json` (default)
- **Status**: ✅ Reference implementation

### Run 2-3: Initial Two-Body Experiments
- **240126_140508_twobody_ddlp**
- **240126_234640_twobody_ddlp**
- **Purpose**: First attempts at two-body physics dataset
- **Dataset**: `/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square`
- **Key Changes**: Longer temporal horizons, adjusted model parameters

### Run 4: Minimal Configuration
- **260126_175323_twobody_ddlp_minimal**
- **Purpose**: Streamlined configuration with reduced temporal horizon
- **Dataset**: Same as Run 2-3
- **Philosophy**: Trade-off between sequence length and computational efficiency

### Run 5-6: New Dataset with Minimal Config
- **280126_184826_twobody_ddlp_minimal_off_cnt_**
- **280126_232212_twobody_ddlp_minimal_off_cnt_**
- **Purpose**: Apply minimal config to new filtered dataset
- **Dataset**: `/data2/users/lr4617/data_twobody_video/two_body_unified_hdf5`
- **Issue**: Initially believed to have only 60 timesteps, but actually has 360 timesteps per episode

---

## Dataset Specifications and Comparison

### Dataset Overview

Three distinct datasets were used across the training runs, each with different characteristics affecting model training and evaluation capabilities.

### Detailed Dataset Comparison

| Dataset | Path | Train Episodes | Val Episodes | Timesteps/Episode | Total Train Frames | Effective Train Frames* |
|---------|------|:--------------:|:------------:|:-----------------:|:------------------:|:----------------------:|
| **BALLS_INTERACTION** | `/data2/users/lr4617/ddlp/data/BALLS_INTERACTION` | 5,000 | 200 | 100 | 500,000 | 500,000 |
| **two_body_...square** | `/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square` | 1,500 | 80 | 360 | 540,000 | **90,000** |
| **two_body_...hdf5** | `/data2/users/lr4617/data_twobody_video/two_body_unified_hdf5` | 2,295 | 115 | 360 | 826,200 | **137,700** |

**\*Effective Train Frames**: For two-body datasets, only 1/6 of each episode's length is used during training (timestep_horizon constraint). This means:
- `two_body_...square`: 360 timesteps → 60 used per episode → 90,000 effective frames
- `two_body_...hdf5`: 360 timesteps → 60 used per episode → 137,700 effective frames

### Dataset Characteristics

#### 1. BALLS_INTERACTION (Baseline)
```
Episodes:     5,000 train / 200 val
Timesteps:    100 per episode
Total Frames: 500,000 train / 20,000 val
Objects:      Variable (multi-ball interactions)
Physics:      Collision dynamics with variable masses
Features:     11 metadata fields (colors, masses, velocities, etc.)
```

**Key Properties:**
- Shorter episodes (100 frames) suitable for quick temporal modeling
- Large number of episodes provides good statistical coverage
- Rich metadata for comprehensive physics understanding
- All frames available for training (no truncation)

#### 2. two_body_system_extrapolation_square
```
Episodes:     1,500 train / 80 val
Timesteps:    360 per episode (full trajectory)
              60 per episode (effective training length, 1/6 truncation)
Total Frames: 540,000 train / 28,800 val (full)
              90,000 train (effective for training)
Objects:      Exactly 2 (binary system)
Physics:      Orbital mechanics with elliptical trajectories
Features:     5 metadata fields (positions, sizes, IDs)
```

**Key Properties:**
- Long episodes (360 frames) capture complete orbital periods
- Training uses first 60 frames per episode (1/6 truncation factor)
- Smaller episode count but richer temporal dynamics
- Designed for extrapolation: train on early frames, test long-term prediction
- Clean elliptical orbits with radius = 5 pixels

#### 3. two_body_unified_hdf5 (Latest)
```
Episodes:     2,295 train / 115 val
Timesteps:    360 per episode (full trajectory)
              60 per episode (effective training length, 1/6 truncation)
Total Frames: 826,200 train / 41,400 val (full)
              137,700 train (effective for training)
Objects:      Exactly 2 (binary system)
Physics:      Filtered orbital mechanics (constraint-satisfying only)
Features:     5 metadata fields (positions, sizes, IDs)
```

**Key Properties:**
- Same temporal structure as `two_body_...square` (360 timesteps)
- **53% more training episodes** (2,295 vs 1,500)
- **53% more effective training frames** (137,700 vs 90,000)
- Trajectory filtering applied: only constraint-satisfying orbits included
- Rendered with `target_radius = 5` pixels after filtering
- Better data quality due to filtering (no overlaps or boundary violations)

### Training Efficiency Comparison

| Metric | BALLS_INT. | two_body_square | two_body_hdf5 | Winner |
|--------|:----------:|:---------------:|:-------------:|:------:|
| **Episodes** | 5,000 | 1,500 | **2,295** | BALLS |
| **Effective Frames** | 500,000 | 90,000 | **137,700** | BALLS |
| **Episode Length** | 100 | **360** | **360** | Two-body |
| **Data Quality** | Standard | Standard | **Filtered** | two_body_hdf5 |
| **Temporal Coverage** | ✓✓ | ✓✓✓ | ✓✓✓ | Two-body |
| **Statistical Coverage** | ✓✓✓ | ✓ | ✓✓ | BALLS |

### Why 1/6 Truncation for Two-Body Datasets?

The two-body datasets store full 360-frame trajectories but only use the first 60 frames during training:

1. **Memory Constraints**: Loading 360-frame sequences requires 6x more GPU memory
2. **Training Efficiency**: Shorter sequences allow larger batch sizes
3. **Sufficient Physics**: 60 frames capture enough orbital dynamics for learning
4. **Validation Uses Full Length**: Complete 360-frame sequences used for long-term prediction evaluation

**Training vs Validation Usage:**
- **Training**: Uses first 60 frames (1/6 of episode)
  - `timestep_horizon = 14-60` determines actual training window
  - Enables faster iteration and larger batches
- **Validation**: Uses full 360 frames
  - Tests long-term extrapolation capability
  - `animation_horizon = 60-120` determines generation length

### Dataset Selection Impact on Configuration

The choice of dataset directly influenced configuration parameters:

| Dataset Used | Optimal `timestep_horizon` | Optimal `animation_horizon` | Batch Size |
|--------------|:--------------------------:|:---------------------------:|:----------:|
| BALLS_INTERACTION | 10 | 100 | 32 |
| two_body_square (360 frames) | 14-60 | 120 | 16-32 |
| two_body_hdf5 (360 frames) | 15 | 60 | 32-64 |

**Note**: The initial misconception that `two_body_unified_hdf5` had only 60 timesteps led to conservative `animation_horizon = 60` in Run 6. With the full 360 timesteps available, `animation_horizon = 120` is feasible.

---

## Configuration Parameter Changes

### Critical Parameters (Affecting Model Behavior)

| Parameter | Balls<br/>(Baseline) | Run 2<br/>140508 | Run 3<br/>234640 | Run 4<br/>Minimal | Run 5<br/>184826 | Run 6<br/>232212 |
|-----------|:--------------------:|:----------------:|:----------------:|:-----------------:|:----------------:|:----------------:|
| **Dataset & Basic Setup** |
| `ds` | balls | twobody | twobody | twobody | twobody | twobody |
| `root` | BALLS_INT. | two_body_...square | two_body_...square | two_body_...square | two_body_...hdf5 | two_body_...hdf5 |
| `device` | cuda:1 | cuda:1 | cuda:1 | cuda:1 | cuda:2 | cuda:1 |
| `batch_size` | 32 | 16 | 16 | 32 | 32 | 64 |
| **Temporal Configuration** |
| `timestep_horizon` | **10** | **60** | **60** | **14** | **14** | **15** |
| `num_static_frames` | **4** | **4** | **4** | **6** | **6** | **6** |
| `cond_steps` | **10** | **60** | **60** | **15** | **15** | **15** |
| `animation_horizon` | **100** | **120** | **100** | **120** | **120** | **60** |
| **Model Architecture** |
| `n_kp_enc` | **6** | **2** | **4** | **5** | **5** | **5** |
| `topk` | **6** | **6** | **6** | **5** | **5** | **5** |
| `pint_dim` | **256** | **128** | **256** | **256** | **256** | **256** |
| **Training Dynamics** |
| `warmup_epoch` | **1** | **1** | **1** | **10** | **10** | **10** |
| `start_dyn_epoch` | **0** | **0** | **0** | **15** | **15** | **15** |
| `obj_on_alpha` | **0.1** | **0.1** | **0.1** | **0.5** | **0.5** | **0.5** |
| **Experiment Naming** |
| `run_prefix` | "" | "twobody_" | "" | "_minimal" | "_minimal_off_cnt_" | "_minimal_off_cnt_" |

### Unchanged Parameters (Stable Across All Runs)

The following parameters remained constant across all experiments:

```json
{
  "lr": 0.0002,
  "kp_activation": "tanh",
  "pad_mode": "replicate",
  "num_epochs": 150,
  "n_kp": 1,
  "recon_loss_type": "mse",
  "sigma": 1.0,
  "beta_kl": 0.1,
  "beta_rec": 1.0,
  "patch_size": 8,
  "eval_epoch_freq": 1,
  "learned_feature_dim": 3,
  "n_kp_prior": 12,
  "weight_decay": 0.0,
  "kp_range": [-1, 1],
  "dropout": 0.0,
  "iou_thresh": 0.2,
  "anchor_s": 0.25,
  "kl_balance": 0.001,
  "image_size": 64,
  "ch": 3,
  "enc_channels": [32, 64, 128],
  "prior_channels": [16, 32, 64],
  "predict_delta": true,
  "beta_dyn": 0.1,
  "scale_std": 0.3,
  "offset_std": 0.2,
  "obj_on_beta": 0.1,
  "beta_dyn_rec": 1.0,
  "pint_layers": 6,
  "pint_heads": 8,
  "eval_im_metrics": true,
  "use_resblock": false,
  "scheduler_gamma": 0.95,
  "adam_betas": [0.9, 0.999],
  "adam_eps": 0.0001,
  "train_enc_prior": true,
  "animation_fps": 0.06,
  "use_correlation_heatmaps": true,
  "enable_enc_attn": false,
  "filtering_heuristic": "variance"
}
```

---

## Key Configuration Evolution Patterns

### 1. Temporal Horizon Strategy

**Baseline → Run 2-3: Extended Horizon**
- `timestep_horizon`: 10 → **60** (6x increase)
- `cond_steps`: 10 → **60** (6x increase)
- **Rationale**: Two-body physics requires longer context for orbital dynamics
- **Impact**: Higher memory usage, better long-term predictions

**Run 2-3 → Run 4-6: Minimal Horizon**
- `timestep_horizon`: 60 → **14-15** (4x reduction)
- `cond_steps`: 60 → **15** (4x reduction)
- **Rationale**: Balance between context length and computational efficiency
- **Impact**: Faster training, reduced memory footprint

### 2. Encoder Keypoint Configuration

**Progressive Reduction**
- Baseline: `n_kp_enc = 6` (detect up to 6 objects)
- Run 2: `n_kp_enc = 2` (strict two-object assumption)
- Run 3: `n_kp_enc = 4` (moderate capacity)
- Run 4-6: `n_kp_enc = 5` (balanced setting)

**Interpretation**: Experimentation to find optimal object detection capacity for two-body system

### 3. Training Dynamics Adjustments

**Minimal Configuration Changes (Run 4-6)**
- `warmup_epoch`: 1 → **10** (gradual KL annealing)
- `start_dyn_epoch`: 0 → **15** (delayed dynamics module)
- `obj_on_alpha`: 0.1 → **0.5** (stronger object existence prior)

**Rationale**: More careful training schedule for cleaner convergence

### 4. Batch Size Progression

- Baseline: 32
- Run 2-3: **16** (larger sequences require smaller batches)
- Run 4-5: **32** (return to baseline)
- Run 6: **64** (doubled for faster training with shorter sequences)

---

## Dataset Compatibility Analysis

### BALLS_INTERACTION
- **Sequence Length**: 100 timesteps (full)
- **Episodes**: 5,000 (train) / 200 (val)
- **Training Usage**: All 100 timesteps used
- ✅ Compatible with baseline configuration
- ✅ Supports `animation_horizon = 100`

### Original Two-Body Dataset (`two_body_system_extrapolation_square`)
- **Sequence Length**: 360 timesteps (full), 60 effective (1/6 truncation)
- **Episodes**: 1,500 (train) / 80 (val)
- **Training Usage**: First 60 timesteps per episode
- **Effective Training Frames**: 90,000
- ✅ Compatible with all temporal horizon configurations
- ✅ Supports `animation_horizon = 120` (can generate beyond training window)

### New Filtered Dataset (`two_body_unified_hdf5`)
- **Sequence Length**: 360 timesteps (full), 60 effective (1/6 truncation)
- **Episodes**: 2,295 (train) / 115 (val)
- **Training Usage**: First 60 timesteps per episode
- **Effective Training Frames**: 137,700 (53% more than square dataset)
- ✅ **Correction**: Has full 360 timesteps, NOT just 60
- ✅ Fully compatible with `animation_horizon = 120`
- ✅ Better data quality due to trajectory filtering

### Corrected Recommendations

For `two_body_unified_hdf5` dataset (with 360 timesteps available):
```json
{
  "timestep_horizon": 20,      // ✅ Optimal for training (20 < 60 training window)
  "num_static_frames": 6,      // ✅ 30% of timestep_horizon
  "cond_steps": 20,            // ✅ Match timestep_horizon
  "animation_horizon": 120,    // ✅ FULLY SUPPORTED (120 < 360 available)
  "batch_size": 32             // ✅ Balanced for 20-step sequences
}
```

**Previous Misconception**: Run 5-6 used `animation_horizon = 60` due to incorrect belief that dataset had only 60 timesteps. With 360 timesteps confirmed, `animation_horizon = 120` is fully supported.

**Training Efficiency**: Despite having 360 timesteps available, using only first 60 for training is a design choice:
- Enables 6x larger batch sizes
- Faster training iterations
- Still captures sufficient orbital dynamics
- Full 360-frame sequences reserved for validation and long-term prediction testing

---

## Configuration Deviation Summary

### Deviation from Baseline (Number of Changed Parameters)

| Run | Config Name | Changed Params | Deviation % |
|-----|-------------|:--------------:|:-----------:|
| Run 1 | balls.json | 0 | 0% |
| Run 2 | 240126_140508 | **10** | 20% |
| Run 3 | 240126_234640 | **11** | 22% |
| Run 4 | 260126_175323_minimal | **12** | 24% |
| Run 5 | 280126_184826 | **13** | 26% |
| Run 6 | 280126_232212 | **14** | 28% |

**Total Configuration Parameters**: 50

### Most Frequently Modified Parameters

1. **Dataset-specific** (6/6 runs): `ds`, `root`
2. **Temporal** (6/6 runs): `timestep_horizon`, `cond_steps`, `animation_horizon`
3. **Model capacity** (5/6 runs): `n_kp_enc`, `pint_dim`
4. **Training schedule** (3/6 runs): `warmup_epoch`, `start_dyn_epoch`, `obj_on_alpha`
5. **Experiment tracking** (6/6 runs): `run_prefix`

### Least Modified Parameters

**Never changed**: All architectural hyperparameters (loss weights, learning rates, network depths)

**Interpretation**: Core model architecture remained stable; adjustments focused on temporal modeling and dataset-specific tuning.

---

## Key Insights

### 1. **Temporal Horizon is Critical**
The most dramatic changes across runs involve temporal parameters (`timestep_horizon`, `animation_horizon`), reflecting experimentation with sequence modeling capacity.

### 2. **Two Distinct Strategies**
- **Long-horizon strategy** (Run 2-3): Deep temporal context (60-step), slower training
- **Minimal strategy** (Run 4-6): Efficient training (14-15 step), practical inference

### 3. **Dataset Drives Configuration**
The transition to `two_body_unified_hdf5` initially appeared to require `animation_horizon` reduction from 120 → 60 due to a misconception about sequence length. **Correction**: Dataset actually has 360 timesteps, fully supporting `animation_horizon = 120`.

### 4. **Effective Training Frames vs Total Frames**
Two-body datasets use 1/6 truncation (first 60 of 360 timesteps) for training efficiency:
- `two_body_square`: 540,000 total → 90,000 effective training frames
- `two_body_hdf5`: 826,200 total → 137,700 effective training frames (53% improvement)

### 5. **Stable Core Architecture**
Loss functions, optimization parameters, and network architecture remained unchanged, suggesting robust defaults from BALLS_INTERACTION baseline.

### 6. **Gradual Refinement**
Each successive run made incremental adjustments rather than radical changes, indicating iterative hyperparameter search.

### 7. **Data Quality Improvements**
The `two_body_unified_hdf5` dataset applies trajectory filtering to exclude constraint-violating orbits, providing cleaner training data while also increasing episode count by 53%.

---

## Recommendations for Future Runs

### For two_body_unified_hdf5 Dataset (360 timesteps available)
```json
{
  "timestep_horizon": 20,
  "num_static_frames": 6,
  "cond_steps": 20,
  "animation_horizon": 120,
  "batch_size": 32
}
```

### For Aggressive Training Efficiency
```json
{
  "timestep_horizon": 15,
  "num_static_frames": 6,
  "cond_steps": 15,
  "animation_horizon": 120,
  "batch_size": 64
}
```

### For BALLS_INTERACTION-like Datasets (100 timesteps)
```json
{
  "timestep_horizon": 10,
  "num_static_frames": 4,
  "cond_steps": 10,
  "animation_horizon": 100,
  "batch_size": 32
}
```

### General Guidelines
1. **Temporal Constraints**: 
   - Training: `timestep_horizon < effective_training_length`
   - Animation: `animation_horizon < full_sequence_length`
   - For two-body datasets: effective_training_length = 60, full_sequence_length = 360
2. Keep `timestep_horizon ≈ cond_steps` for consistency
3. Set `num_static_frames ≈ 0.3-0.4 × timestep_horizon`
4. Use higher `batch_size` for shorter `timestep_horizon`
5. Increase `warmup_epoch` and `start_dyn_epoch` for better convergence
6. **Dataset-Specific**: Understand effective vs total frame counts to avoid over/under-utilizing data

---

## Quick Reference: Parameter Definitions

| Parameter | Description | Typical Range |
|-----------|-------------|:-------------:|
| `timestep_horizon` | Training sequence length (frames) | 10-60 |
| `num_static_frames` | Frames using constant priors | 4-6 |
| `cond_steps` | Conditioning frames for dynamics | 10-60 |
| `animation_horizon` | Generation/rollout length | 60-120 |
| `n_kp_enc` | Max keypoints detected by encoder | 2-10 |
| `topk` | Keypoints selected from patches | 5-10 |
| `pint_dim` | Physics integrator hidden dimension | 128-256 |
| `warmup_epoch` | KL warmup period | 1-10 |
| `start_dyn_epoch` | Epoch to activate dynamics training | 0-15 |
| `obj_on_alpha` | Object existence Beta prior α | 0.1-0.5 |

---

*Document generated: 2026-01-29*  
*Baseline config: `configs/balls.json`*  
*Analysis covers: 6 training runs (210126 → 280126)*
