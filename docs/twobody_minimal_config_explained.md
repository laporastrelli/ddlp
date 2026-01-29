# Two-Body Dataset Configuration Parameters

This document explains each parameter in `twobody_minimal.json` in terms of its role in the code and the underlying DDLP model formalism.

---

## Dataset Parameters

### `ds`: "twobody"
- **Code Role**: Dataset identifier used by `get_dataset.py` to load the appropriate dataset class (`TwoBodyDataset`)
- **Model Formalism**: N/A (infrastructure)

### `root`: "/data2/users/lr4617/data/ddlp/two_body_system_extrapolation_square"
- **Code Role**: Path to HDF5 files containing training/validation/test data
- **Model Formalism**: N/A (infrastructure)

### `device`: "cuda:1"
- **Code Role**: GPU device for training (uses second GPU)
- **Model Formalism**: N/A (infrastructure)

---

## Training Hyperparameters

### `batch_size`: 32
- **Code Role**: Number of video sequences processed simultaneously per training iteration
- **Model Formalism**: Batch dimension B in tensor shapes [B, T, C, H, W]
- **Trade-off**: Larger batch → more stable gradients but higher memory usage

### `lr`: 0.0002
- **Code Role**: Initial learning rate for Adam optimizer
- **Model Formalism**: Step size α in gradient descent: θ_{t+1} = θ_t - α∇L(θ_t)
- **Value**: 2×10^-4 is standard for Adam with β=(0.9, 0.999)

### `num_epochs`: 150
- **Code Role**: Total training epochs (one pass through entire training set)
- **Model Formalism**: N/A (training schedule)

### `scheduler_gamma`: 0.95
- **Code Role**: Learning rate decay factor applied per epoch: lr_{t+1} = γ × lr_t
- **Model Formalism**: Implements learning rate schedule for convergence
- **Effect**: lr decreases by 5% each epoch (lr at epoch 150 ≈ 0.0002 × 0.95^150 ≈ 1.4×10^-5)

### `adam_betas`: [0.9, 0.999]
- **Code Role**: Exponential decay rates for Adam's moment estimates
- **Model Formalism**: 
  - β₁ = 0.9: first moment (mean) decay rate
  - β₂ = 0.999: second moment (variance) decay rate
  - m_t = β₁m_{t-1} + (1-β₁)g_t
  - v_t = β₂v_{t-1} + (1-β₂)g_t²

### `adam_eps`: 0.0001
- **Code Role**: Epsilon term for numerical stability in Adam
- **Model Formalism**: θ_{t+1} = θ_t - α·m̂_t/(√v̂_t + ε)
- **Value**: 1×10^-4 prevents division by zero

### `weight_decay`: 0.0
- **Code Role**: L2 regularization coefficient
- **Model Formalism**: Adds λ||θ||² to loss (disabled when 0)

---

## Model Architecture

### `image_size`: 64
- **Code Role**: Input/output image resolution (64×64 pixels)
- **Model Formalism**: Spatial dimensions H=W=64 in x ∈ ℝ^{B×T×3×64×64}

### `ch`: 3
- **Code Role**: Number of color channels (RGB)
- **Model Formalism**: Channel dimension C=3 in x ∈ ℝ^{B×T×3×H×W}

### `enc_channels`: [32, 64, 128]
- **Code Role**: CNN encoder channel progression across layers
- **Model Formalism**: Feature map dimensions in posterior encoder q(z|x)
- **Architecture**: 
  - Layer 1: 3→32 channels
  - Layer 2: 32→64 channels  
  - Layer 3: 64→128 channels
  - Output: 128-dimensional feature maps

### `prior_channels`: [16, 32, 64]
- **Code Role**: CNN prior encoder channel progression
- **Model Formalism**: Feature map dimensions in prior p(z)
- **Note**: Half the capacity of posterior encoder (faster, less expressive)

### `pint_dim`: 256
- **Code Role**: Transformer hidden dimension for dynamics model
- **Model Formalism**: Embedding dimension d_model in PINT transformer
- **Usage**: Each particle's state is projected to 256-dim before self-attention

### `pint_layers`: 6
- **Code Role**: Number of transformer layers in PINT dynamics module
- **Model Formalism**: Depth of p(z_{t+1}|z_t) dynamics predictor
- **Trade-off**: More layers → more expressive but slower and prone to overfitting

### `pint_heads`: 8
- **Code Role**: Number of attention heads in PINT transformer
- **Model Formalism**: Multi-head attention with h=8 heads
- **Per-head dimension**: d_k = d_model / h = 256 / 8 = 32

### `use_resblock`: false
- **Code Role**: Whether to use residual blocks in CNN encoder/decoder
- **Model Formalism**: Adds skip connections: y = f(x) + x (disabled)

---

## Keypoint Detection Parameters

### `n_kp_prior`: 12
- **Code Role**: Number of prior keypoint proposals from CNN
- **Model Formalism**: K_prior keypoints from prior network before filtering
- **Usage**: Initial proposals covering spatial grid (64 patches → top 12)

### `n_kp_enc`: 5
- **Code Role**: Number of posterior particles (object representations)
- **Model Formalism**: K particles in posterior q(z₁,...,z_K|x)
- **Reasoning**: 5 particles for 2 objects (redundancy for discovery + background)

### `topk`: 5
- **Code Role**: Number of top prior proposals to keep after filtering
- **Model Formalism**: Selects K particles with highest variance/score
- **Note**: Must equal n_kp_enc (topk → posterior particles)

### `filtering_heuristic`: "variance"
- **Code Role**: Method to select top-k prior proposals
- **Model Formalism**: Ranks proposals by feature variance σ²
- **Alternatives**: "none" (no filtering), "random", "distance"

### `n_kp`: 1
- **Code Role**: Gaussian components per particle (always 1 in DDLP)
- **Model Formalism**: Each particle has 1 spatial Gaussian N(μ, σ²)

### `kp_activation`: "tanh"
- **Code Role**: Activation function for keypoint positions
- **Model Formalism**: μ = tanh(logits) ∈ [-1, 1] (normalized coordinates)

### `kp_range`: [-1, 1]
- **Code Role**: Valid range for keypoint coordinates
- **Model Formalism**: Spatial domain for μ ∈ [-1,1]² (image corners at ±1)

### `sigma`: 1.0
- **Code Role**: Standard deviation of Gaussian heatmaps for keypoints
- **Model Formalism**: σ in Gaussian G(x,y|μ,σ²) for spatial attention
- **Effect**: Larger σ → wider attention region, easier object discovery

### `patch_size`: 8
- **Code Role**: Size of image patches for prior proposals (8×8 pixels)
- **Model Formalism**: Local receptive field for initial keypoint detection
- **Computation**: 64×64 image → 8×8 grid of patches (64 proposals total)

---

## Object Representation Parameters

### `learned_feature_dim`: 3
- **Code Role**: Dimensionality of learned visual features per particle
- **Model Formalism**: z_feat ∈ ℝ³ for each particle (appearance embedding)
- **Usage**: Encodes object appearance (color, texture) beyond position

### `anchor_s`: 0.25
- **Code Role**: Default/prior value for object scale
- **Model Formalism**: Prior mean for z_scale (log-odds space)
- **Interpretation**: Objects expected to be ~25% of image size initially

### `scale_std`: 0.3
- **Code Role**: Standard deviation for scale prior p(z_scale)
- **Model Formalism**: σ_scale in N(logit(anchor_s), σ²_scale)

### `offset_std`: 0.2
- **Code Role**: Standard deviation for position refinement offset
- **Model Formalism**: σ_offset in q(μ_offset|x), allows ±0.2 adjustment from prior

---

## Object-On Detection

### `obj_on_alpha`: 0.5
- **Code Role**: Alpha parameter for Beta prior on obj_on probability
- **Model Formalism**: p(z_on) = Beta(α, β) with α=0.5
- **Effect**: 
  - α > β → biased toward obj_on=1 (object present)
  - α = β = 0.1 (balls.json) → uniform prior
  - α = 0.5 > 0.1 → mild bias toward detecting objects

### `obj_on_beta`: 0.1
- **Code Role**: Beta parameter for Beta prior on obj_on probability
- **Model Formalism**: p(z_on) = Beta(0.5, 0.1)
- **Distribution**: Mean = α/(α+β) = 0.5/0.6 ≈ 0.83 (expectation toward ON)

### `iou_thresh`: 0.2
- **Code Role**: IoU threshold for considering particles as duplicates
- **Model Formalism**: Two particles are "same object" if IoU(bbox₁, bbox₂) > 0.2
- **Usage**: Evaluation metric, not used in loss

---

## Training Stages

### `warmup_epoch`: 10
- **Code Role**: Number of epochs with frozen position parameters
- **Model Formalism**: For epochs < 10:
  - Gradients blocked: ∇_{z,μ,scale} = 0
  - Only features/obj_on train: ∇_{z_feat, z_on} ≠ 0
- **Purpose**: Stabilize position learning before feature learning interferes
- **Timeline**:
  - Epochs 0-9: Positions frozen, features train
  - Epochs 10-11: All parameters train + alpha noise
  - Epochs 12+: Normal training

### `start_dyn_epoch`: 15
- **Code Role**: Epoch to start training dynamics module (PINT)
- **Model Formalism**: For epochs < 15:
  - Only static reconstruction: L = L_recon + L_KL_static
  - Epochs ≥ 15: Add dynamics: L = L_recon + L_KL_static + L_dyn
- **Purpose**: Learn good object representations before predicting motion
- **Reasoning**: Dynamics prediction requires stable object encodings

### `num_static_frames`: 6
- **Code Role**: Number of initial frames encoded independently (no dynamics)
- **Model Formalism**: For t ∈ [0, 6): use q(z_t|x_t) (independent encoding)
- **Purpose**: Provides initial "memory" for dynamics module
- **For t ≥ 6**: Use p(z_t|z_{<t}) (dynamics prediction)

---

## Loss Function Weights

### `beta_rec`: 1.0
- **Code Role**: Weight for reconstruction loss
- **Model Formalism**: L = β_rec · L_recon + β_KL · L_KL + β_dyn · L_dyn
- **Typical**: β_rec = 1.0 (base scale for other terms)

### `beta_kl`: 0.1
- **Code Role**: Global weight for KL divergence terms
- **Model Formalism**: β_KL in β-VAE: ELBO = 𝔼[log p(x|z)] - β·KL[q(z|x)||p(z)]
- **Effect**: β_KL = 0.1 → prioritize reconstruction over KL regularization

### `kl_balance`: 0.001
- **Code Role**: Additional scaling for specific KL terms (kp, feat, scale, depth, obj_on)
- **Model Formalism**: 
  - L_KL = kl_balance · (KL_kp + KL_feat + KL_scale + KL_depth + KL_obj_on)
  - Total weight: β_KL · kl_balance = 0.1 · 0.001 = 0.0001
- **Purpose**: Heavily downweight KL penalties to prevent collapse

### `beta_dyn`: 0.1
- **Code Role**: Weight for dynamics prediction loss
- **Model Formalism**: L_dyn = β_dyn · 𝔼[KL[q(z_{t+1}|x_{t+1}) || p(z_{t+1}|z_t)]]
- **Effect**: Dynamics loss weighted equally to static KL

### `beta_dyn_rec`: 1.0
- **Code Role**: Weight for reconstruction loss on dynamically predicted frames
- **Model Formalism**: L_recon_dyn = β_dyn_rec · ||x_t - x̂_t||²
- **Usage**: Same weight as static reconstruction

---

## Sequence Parameters

### `timestep_horizon`: 14
- **Code Role**: Number of frames in training sequences (excluding initial frame)
- **Model Formalism**: T=14 timesteps for dynamics prediction
- **Actual frames**: 15 (1 conditioning + 14 prediction steps)
- **Dataset**: Creates subsequences of length 15 from 60-frame episodes

### `cond_steps`: 15
- **Code Role**: Number of conditioning frames for autoregressive generation
- **Model Formalism**: For generation: encode z_{0:15}, then predict z_{16:T}
- **Note**: cond_steps=15 means use all 15 frames as context (no prediction during training)
- **Evaluation**: For animation, cond_steps < timestep_horizon enables forecasting

### `animation_horizon`: 120
- **Code Role**: Maximum sequence length for evaluation/animation
- **Model Formalism**: T_eval = 120 for long-range forecasting tests
- **Usage**: Validation animations show 120-frame rollouts

### `predict_delta`: true
- **Code Role**: Predict position changes Δμ instead of absolute positions
- **Model Formalism**: 
  - If true: z_t = z_{t-1} + Δz (residual prediction)
  - If false: z_t = f(z_{t-1}) (absolute prediction)
- **Benefit**: Easier to learn small motions

---

## Reconstruction Loss

### `recon_loss_type`: "mse"
- **Code Role**: Type of reconstruction loss function
- **Model Formalism**: 
  - "mse": L_recon = ||x - x̂||²₂ (L2 norm)
  - Alternatives: "l1", "vgg" (perceptual loss)
- **Standard**: MSE for VAE-based models

### `pad_mode`: "replicate"
- **Code Role**: Padding mode for spatial transformer (STN)
- **Model Formalism**: How to handle out-of-bounds coordinates in affine transforms
- **Options**: "replicate" (repeat edge pixels), "zeros", "border"

---

## Training Flags

### `train_enc_prior`: true
- **Code Role**: Whether to train prior encoder p(z)
- **Model Formalism**: If true: ∇_θ_prior L; If false: θ_prior frozen
- **Usage**: Typically true (both posterior and prior learn)

### `enable_enc_attn`: false
- **Code Role**: Enable self-attention in CNN encoder
- **Model Formalism**: Adds attention layers to q(z|x) encoder (disabled for speed)

### `dropout`: 0.0
- **Code Role**: Dropout probability in neural networks
- **Model Formalism**: Randomly zero neurons with probability p=0.0 (disabled)

---

## Visualization Parameters

### `animation_fps`: 0.06
- **Code Role**: Frames per second for saved animation videos (1/fps = 16.67s per frame)
- **Model Formalism**: N/A (visualization)
- **Note**: Very slow playback for detailed inspection

### `use_correlation_heatmaps`: true
- **Code Role**: Visualize spatial attention via correlation heatmaps
- **Model Formalism**: Shows Gaussian heatmaps G(x,y|μ,σ²) overlaid on images

### `eval_im_metrics`: true
- **Code Role**: Compute image metrics (PSNR, SSIM, LPIPS) during validation
- **Model Formalism**: N/A (evaluation)

### `eval_epoch_freq`: 1
- **Code Role**: Run validation every N epochs
- **Model Formalism**: N/A (training schedule)

---

## Model Identification

### `run_prefix`: "_minimal"
- **Code Role**: Suffix appended to experiment directory name
- **Model Formalism**: N/A (bookkeeping)
- **Example**: Creates folder "YYMMDD_HHMMSS_twobody_ddlp_minimal/"

### `load_model`: false
- **Code Role**: Whether to load pretrained checkpoint
- **Model Formalism**: N/A (training initialization)

### `pretrained_path`: null
- **Code Role**: Path to checkpoint file for loading
- **Model Formalism**: N/A (training initialization)

---

## Summary: Key Differences from balls.json

| Parameter | balls.json | twobody_minimal.json | Rationale |
|-----------|------------|----------------------|-----------|
| `n_kp_enc` / `topk` | 6 | 5 | Fewer particles for 2 objects vs 3 |
| `obj_on_alpha` | 0.1 | 0.5 | Mild bias to prevent obj_on collapse |
| `warmup_epoch` | 1 | 10 | Longer stabilization for limited data |
| `start_dyn_epoch` | 0 | 15 | Delayed dynamics to learn good features first |
| `num_static_frames` | 4 | 6 | More context frames for dynamics |
| `timestep_horizon` | 10 | 14 | Longer sequences for richer dynamics |

**Philosophy**: Start with balls.json as baseline, make minimal adjustments to account for:
1. Simpler scene (2 vs 3 objects) → fewer particles
2. Data scarcity (10x less data) → longer warmup, delayed dynamics
3. obj_on collapse issue → slightly biased prior (α=0.5 vs 0.1)

## Possible Causes of Failure
- **background**: From the paper description, the background latent particle "is always located in the center of the image". This means that for this particular dataset, where the position of the two bodies in the scene is always initialized at the centre of the frame, the background latent particle will probably capture parts of the bodies information as well.
