# Minimal and Intermediate Latent Representations for DDLP

## Overview

This document describes the modifications needed to implement reduced latent representations for the Balls-Interactions dataset, which is simpler than datasets with occlusions and 3D depth.

### Current (Full) Representation: 9D per particle
- **z_p** (position): 2D - (x, y) coordinates
- **z_s** (scale): 2D - scale in x and y directions
- **z_t** (transparency/obj_on): 1D - object existence probability
- **z_d** (depth): 1D - z-ordering for occlusions
- **z_f** (features): 3D - visual appearance features

**Total: 2 + 2 + 1 + 1 + 3 = 9 dimensions**

### Intermediate Representation: 7D per particle
- **z_p** (position): 2D ✓
- **z_s** (scale): 2D ✓
- **z_f** (features): 3D ✓
- ~~z_t (transparency)~~: Removed - objects never disappear
- ~~z_d (depth)~~: Removed - no occlusions in 2D

**Total: 2 + 2 + 3 = 7 dimensions** (~22% reduction)

### Minimal Representation: 5D per particle
- **z_p** (position): 2D ✓
- **z_f** (features): 3D ✓
- ~~z_s (scale)~~: Removed - all balls have same size
- ~~z_t (transparency)~~: Removed - objects never disappear
- ~~z_d (depth)~~: Removed - no occlusions in 2D

**Total: 2 + 3 = 5 dimensions** (~44% reduction)

---

## Why Config Changes Alone Are Not Sufficient

The DDLP architecture has **hard-coded neural network heads** for all dimensions:
- `depth_head = nn.Linear(hidden_dim_2, 2)`
- `obj_on_head = nn.Linear(hidden_dim_2, 2)`
- `scale_xy_head = nn.Linear(hidden_dim_2, 4)`

These heads are always initialized and used in forward passes, regardless of config settings. The architecture must be modified to support reduced representations.

---

## Configuration Files

Configuration files have been created for each representation:

### 1. `configs/balls.json` (Original - 9D)
Standard configuration with all dimensions.

### 2. `configs/balls_intermediate.json` (7D)
```json
{
  ...
  "run_prefix": "intermediate_",
  "use_depth": false,
  "use_transparency": false,
  "_comment": "z_p(2D) + z_s(2D) + z_f(3D) = 7D per particle"
}
```

### 3. `configs/balls_minimal.json` (5D)
```json
{
  ...
  "run_prefix": "minimal_",
  "use_depth": false,
  "use_transparency": false,
  "use_scale": false,
  "fixed_scale_value": 0.25,
  "_comment": "z_p(2D) + z_f(3D) = 5D per particle"
}
```

---

## Required Code Modifications

### 1. `modules/modules.py` - ParticleAttributeEncoder

**Location:** Lines ~703-850

#### A. Add configuration parameters to `__init__`

```python
class ParticleAttributeEncoder(nn.Module):
    def __init__(self, anchor_size, image_size, cnn_channels=(16, 16, 32), 
                 margin=0, ch=3, max_offset=1.0, kp_activation='tanh', 
                 use_resblock=False, use_correlation_heatmaps=False, 
                 enable_attn=False, hidden_dims=(256, 256), attn_dropout=0.1,
                 use_depth=True, use_transparency=True, use_scale=True):  # ADD THESE
        super().__init__()
        # ... existing code ...
        
        self.use_depth = use_depth
        self.use_transparency = use_transparency
        self.use_scale = use_scale
        
        # Existing heads
        self.x_head = nn.Linear(hidden_dim_2, 2)
        self.y_head = nn.Linear(hidden_dim_2, 2)
        
        # Conditional heads
        if self.use_scale:
            self.scale_xy_head = nn.Linear(hidden_dim_2, 4)
        if self.use_transparency:
            self.obj_on_head = nn.Linear(hidden_dim_2, 2)
        if self.use_depth:
            self.depth_head = nn.Linear(hidden_dim_2, 2)
```

#### B. Modify `forward()` method

```python
def forward(self, x, kp, z_scale=None, previous_objects=None):
    # ... existing code up to backbone_features ...
    
    # Always compute position (xy)
    stats_x = self.x_head(backbone_features)
    stats_x = stats_x.view(batch_size, n_kp, 2)
    mu_x, logvar_x = stats_x.chunk(chunks=2, dim=-1)
    
    stats_y = self.y_head(backbone_features)
    stats_y = stats_y.view(batch_size, n_kp, 2)
    mu_y, logvar_y = stats_y.chunk(chunks=2, dim=-1)
    
    mu = torch.cat([mu_x, mu_y], dim=-1)
    logvar = torch.cat([logvar_x, logvar_y], dim=-1)
    
    # Conditional computations
    if self.use_scale:
        scale_xy = self.scale_xy_head(backbone_features)
        scale_xy = scale_xy.view(batch_size, n_kp, -1)
        mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)
    else:
        # Return None or use fixed scale from config
        mu_scale = None
        logvar_scale = None
    
    if self.use_transparency:
        obj_on = self.obj_on_head(backbone_features)
        obj_on = obj_on.view(batch_size, n_kp, 2)
        lobj_on_a, lobj_on_b = torch.chunk(obj_on, chunks=2, dim=-1)
        lobj_on_a = lobj_on_a.squeeze(-1)
        lobj_on_b = lobj_on_b.squeeze(-1)
    else:
        # Objects always "on" - set to high confidence values
        lobj_on_a = torch.ones(batch_size, n_kp, device=x.device) * 2.0
        lobj_on_b = torch.ones(batch_size, n_kp, device=x.device) * 0.1
        obj_on = torch.stack([lobj_on_a, lobj_on_b], dim=-1)
    
    if self.use_depth:
        depth = self.depth_head(backbone_features)
        depth = depth.view(batch_size, n_kp, 2)
        mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
    else:
        # Fixed depth (all at same level)
        mu_depth = torch.zeros(batch_size, n_kp, 1, device=x.device)
        logvar_depth = torch.zeros(batch_size, n_kp, 1, device=x.device) - 10.0  # low variance
    
    spatial_out = {
        'mu': mu, 'logvar': logvar, 
        'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
        'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b, 'obj_on': obj_on,
        'mu_depth': mu_depth, 'logvar_depth': logvar_depth
    }
    return spatial_out
```

---

### 2. `modules/dynamics_modules.py` - ParticleFeatureProjection

**Location:** Lines ~254-315

#### A. Modify `__init__` to handle different dimensions

```python
class ParticleFeatureProjection(torch.nn.Module):
    def __init__(self, in_features_dim, bg_features_dim, hidden_dim, output_dim,
                 use_depth=True, use_transparency=True, use_scale=True):  # ADD THESE
        super().__init__()
        self.in_features_dim = in_features_dim
        self.bg_features_dim = bg_features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_depth = use_depth
        self.use_transparency = use_transparency
        self.use_scale = use_scale
        
        # Calculate particle dimension based on active components
        self.particle_dim = 2  # position (always)
        if use_scale:
            self.particle_dim += 2
        if use_transparency:
            self.particle_dim += 1
        if use_depth:
            self.particle_dim += 1
        self.particle_dim += in_features_dim  # features (always)
        
        # Projections
        self.xy_projection = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 2)
        )
        
        if use_scale:
            self.scale_projection = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 2)
            )
        if use_transparency:
            self.obj_on_projection = nn.Sequential(
                nn.Linear(1, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 1)
            )
        if use_depth:
            self.depth_projection = nn.Sequential(
                nn.Linear(1, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 1)
            )
        
        self.features_projection = nn.Sequential(
            nn.Linear(in_features_dim, hidden_dim), nn.ReLU(True), 
            nn.Linear(hidden_dim, in_features_dim)
        )
        self.particle_projection = nn.Sequential(
            nn.Linear(self.particle_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.bg_features_projection = nn.Sequential(
            nn.Linear(bg_features_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim)
        )
```

#### B. Modify `forward()` method

```python
def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features):
    bs, n_particles, feat_dim = z_features.shape
    
    # Always project position and features
    z_proj = self.xy_projection(z)
    z_features_proj = self.features_projection(z_features)
    
    # Conditionally project other components
    components = [z_proj]
    
    if self.use_scale:
        z_scale_proj = self.scale_projection(z_scale)
        components.append(z_scale_proj)
    
    if self.use_transparency:
        if len(z_obj_on.shape) == 2:
            z_obj_on = z_obj_on.unsqueeze(-1)
        z_obj_on_proj = self.obj_on_projection(z_obj_on)
        components.append(z_obj_on_proj)
    
    if self.use_depth:
        z_depth_proj = self.depth_projection(z_depth)
        components.append(z_depth_proj)
    
    components.append(z_features_proj)
    
    z_all = torch.cat(components, dim=-1)
    z_all_proj = self.particle_projection(z_all)
    z_bg_features_proj = self.bg_features_projection(z_bg_features)
    z_processed = torch.cat([z_all_proj, z_bg_features_proj.unsqueeze(1)], dim=1)
    
    return z_processed
```

---

### 3. `modules/dynamics_modules.py` - ParticleFeatureDecoder

**Location:** Lines ~317-410

#### A. Modify `__init__`

```python
class ParticleFeatureDecoder(nn.Module):
    def __init__(self, input_dim, features_dim, bg_features_dim, hidden_dim, 
                 kp_activation='tanh', max_delta=1.0, delta_features=False,
                 use_depth=True, use_transparency=True, use_scale=True):  # ADD THESE
        super().__init__()
        self.input_dim = input_dim
        self.features_dim = features_dim
        self.bg_features_dim = bg_features_dim
        self.kp_activation = kp_activation
        self.max_delta = max_delta
        self.delta_features = delta_features
        self.use_depth = use_depth
        self.use_transparency = use_transparency
        self.use_scale = use_scale
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)
        )
        
        # Always have position heads
        self.x_head = nn.Linear(hidden_dim, 2)
        self.y_head = nn.Linear(hidden_dim, 2)
        
        # Conditional heads
        if use_scale:
            self.scale_xy_head = nn.Linear(hidden_dim, 4)
        if use_transparency:
            self.obj_on_head = nn.Linear(hidden_dim, 2)
        if use_depth:
            self.depth_head = nn.Linear(hidden_dim, 2)
        
        self.features_head = nn.Linear(hidden_dim, 2 * features_dim)
        self.bg_features_head = nn.Linear(hidden_dim, 2 * bg_features_dim)
```

#### B. Modify `forward()` method

```python
def forward(self, x):
    bs, n_particles, in_dim = x.shape
    backbone_features = self.backbone(x)
    fg_features, bg_features = backbone_features.split([n_particles - 1, 1], dim=1)
    
    # Always compute position
    stats_x = self.x_head(fg_features)
    stats_x = stats_x.view(bs, n_particles - 1, 2)
    mu_x, logvar_x = stats_x.chunk(chunks=2, dim=-1)
    
    stats_y = self.y_head(fg_features)
    stats_y = stats_y.view(bs, n_particles - 1, 2)
    mu_y, logvar_y = stats_y.chunk(chunks=2, dim=-1)
    
    mu = torch.cat([mu_x, mu_y], dim=-1)
    logvar = torch.cat([logvar_x, logvar_y], dim=-1)
    
    if self.kp_activation == "tanh":
        mu = torch.tanh(mu)
    elif self.kp_activation == "sigmoid":
        mu = torch.sigmoid(mu)
    mu = self.max_delta * mu
    
    # Conditional computations
    if self.use_scale:
        scale_xy = self.scale_xy_head(fg_features)
        scale_xy = scale_xy.view(bs, n_particles - 1, -1)
        mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)
    else:
        mu_scale = None
        logvar_scale = None
    
    if self.use_transparency:
        obj_on = self.obj_on_head(fg_features)
        obj_on = obj_on.view(bs, n_particles - 1, 2)
        lobj_on_a, lobj_on_b = torch.chunk(obj_on, chunks=2, dim=-1)
    else:
        lobj_on_a = torch.ones(bs, n_particles - 1, 1, device=x.device) * 2.0
        lobj_on_b = torch.ones(bs, n_particles - 1, 1, device=x.device) * 0.1
        obj_on = torch.cat([lobj_on_a, lobj_on_b], dim=-1)
    
    if self.use_depth:
        depth = self.depth_head(fg_features)
        depth = depth.view(bs, n_particles - 1, 2)
        mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
    else:
        mu_depth = torch.zeros(bs, n_particles - 1, 1, device=x.device)
        logvar_depth = torch.zeros(bs, n_particles - 1, 1, device=x.device) - 10.0
    
    features = self.features_head(fg_features)
    features = features.view(bs, n_particles - 1, 2 * self.features_dim)
    mu_features, logvar_features = torch.chunk(features, 2, dim=-1)
    
    bg_features = self.bg_features_head(bg_features.squeeze(1))
    mu_bg_features, logvar_bg_features = torch.chunk(bg_features, 2, dim=-1)
    
    decoder_out = {
        'mu': mu, 'logvar': logvar, 
        'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b, 'obj_on': obj_on,
        'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
        'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 
        'mu_features': mu_features, 'logvar_features': logvar_features,
        'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features
    }
    return decoder_out
```

---

### 4. `modules/dynamics_modules.py` - DynamicsDLP

**Location:** Lines ~412-614

Modify initialization to pass flags to sub-modules:

```python
class DynamicsDLP(nn.Module):
    def __init__(self, learned_feature_dim, bg_learned_feature_dim, hidden_dim, 
                 projection_dim, n_head=8, n_layer=6, block_size=20, 
                 kp_activation='tanh', predict_delta=True, max_delta=1.0,
                 positional_bias=False, max_particles=None,
                 use_depth=True, use_transparency=True, use_scale=True):  # ADD THESE
        super().__init__()
        
        self.use_depth = use_depth
        self.use_transparency = use_transparency
        self.use_scale = use_scale
        
        # Pass flags to projection and decoder
        self.particle_projection = ParticleFeatureProjection(
            learned_feature_dim, bg_learned_feature_dim, hidden_dim, projection_dim,
            use_depth=use_depth, use_transparency=use_transparency, use_scale=use_scale
        )
        
        self.particle_transformer = ParticleTransformer(
            n_embed=projection_dim, n_head=n_head, n_layer=n_layer, 
            block_size=block_size, output_dim=projection_dim,
            positional_bias=positional_bias, max_particles=max_particles
        )
        
        self.particle_decoder = ParticleFeatureDecoder(
            projection_dim, learned_feature_dim, bg_learned_feature_dim, 
            hidden_dim, kp_activation=kp_activation, max_delta=max_delta,
            use_depth=use_depth, use_transparency=use_transparency, use_scale=use_scale
        )
```

---

### 5. `models.py` - Loss Computations

**Location:** Lines ~1500-1650 (in ObjectDLP and ObjectDynamicsDLP)

Modify loss computation functions to skip disabled dimensions:

```python
# In calc_fg_loss() or similar loss computation methods:

# KL loss for scale (conditional)
if self.use_scale:
    loss_kl_scale = calc_kl(logvar_scale.view(-1, logvar_scale.shape[-1]),
                            mu_scale.view(-1, mu_scale.shape[-1]), 
                            mu_o=mu_scale_prior, logvar_o=logvar_scale_p, reduce='none')
    loss_kl_scale = (loss_kl_scale.view(-1, self.n_kp_enc)).sum(-1).mean()
else:
    loss_kl_scale = torch.tensor(0.0, device=x.device)

# KL loss for depth (conditional)
if self.use_depth:
    loss_kl_depth = calc_kl(logvar_depth.view(-1, logvar_depth.shape[-1]),
                            mu_depth.view(-1, mu_depth.shape[-1]), reduce='none')
    loss_kl_depth = (loss_kl_depth.view(-1, self.n_kp_enc)).sum(-1).mean()
else:
    loss_kl_depth = torch.tensor(0.0, device=x.device)

# KL loss for transparency (conditional)
if self.use_transparency:
    loss_kl_obj_on = calc_kl_beta_dist(obj_on_a, obj_on_b, 
                                       obj_on_a_prior, obj_on_b_prior, reduce='none')
    loss_kl_obj_on = loss_kl_obj_on.mean()
else:
    loss_kl_obj_on = torch.tensor(0.0, device=x.device)
```

---

### 6. Model Initialization in Training Scripts

**Location:** `train_ddlp.py`, `train_dlp.py`

Add logic to read config flags and pass to model:

```python
# In train_ddlp.py or train_dlp.py

def main():
    # Load config
    config = load_config(args.config_file)
    
    # Read dimension flags (with defaults)
    use_depth = config.get('use_depth', True)
    use_transparency = config.get('use_transparency', True)
    use_scale = config.get('use_scale', True)
    fixed_scale_value = config.get('fixed_scale_value', 0.25)
    
    # Create model with flags
    model = ObjectDynamicsDLP(
        cdim=config['ch'],
        enc_channels=config['enc_channels'],
        prior_channels=config['prior_channels'],
        image_size=config['image_size'],
        n_kp=config['n_kp'],
        n_kp_enc=config['n_kp_enc'],
        n_kp_prior=config['n_kp_prior'],
        learned_feature_dim=config['learned_feature_dim'],
        # ... other params ...
        use_depth=use_depth,
        use_transparency=use_transparency,
        use_scale=use_scale,
    )
    
    # Store in model for later use
    model.use_depth = use_depth
    model.use_transparency = use_transparency
    model.use_scale = use_scale
    model.fixed_scale_value = fixed_scale_value if not use_scale else None
```

---

## Implementation Checklist

- [ ] Modify `ParticleAttributeEncoder` in `modules/modules.py`
- [ ] Modify `ParticleFeatureProjection` in `modules/dynamics_modules.py`
- [ ] Modify `ParticleFeatureDecoder` in `modules/dynamics_modules.py`
- [ ] Modify `DynamicsDLP` in `modules/dynamics_modules.py`
- [ ] Update loss computations in `models.py` (FgDLP, ObjectDLP, ObjectDynamicsDLP)
- [ ] Update training scripts to read and pass config flags
- [ ] Test forward pass with intermediate config
- [ ] Test forward pass with minimal config
- [ ] Test backward pass and gradient flow
- [ ] Test checkpoint saving/loading
- [ ] Verify training runs without errors

---

## Expected Benefits

### Intermediate (7D) - Removes depth and transparency
- **Parameter reduction:** ~22% fewer parameters in attribute heads
- **Training speed:** Faster convergence due to fewer dimensions
- **Suitable for:** 2D datasets without occlusions (Balls, simple Shapes)

### Minimal (5D) - Removes depth, transparency, and scale
- **Parameter reduction:** ~44% fewer parameters in attribute heads
- **Training speed:** Significantly faster due to minimal latent space
- **Memory usage:** Lower memory footprint
- **Suitable for:** Highly constrained datasets (Balls with uniform sizes)

---

## Testing & Validation

### Unit Tests
1. Initialize models with each config
2. Verify forward pass produces correct output shapes
3. Check that disabled dimensions return appropriate default values
4. Verify loss computation handles missing dimensions

### Integration Tests
1. Train for a few epochs with intermediate config
2. Train for a few epochs with minimal config
3. Compare reconstruction quality
4. Verify video prediction works
5. Check that checkpoints save/load correctly

### Performance Tests
1. Measure training time per epoch
2. Measure memory usage
3. Compare final metrics (PSNR, SSIM, FVD)
4. Analyze learned representations

---

## Troubleshooting

### Issue: Model fails to initialize
- **Check:** All modules have the new parameters in `__init__`
- **Check:** Config flags are being read correctly

### Issue: Forward pass errors
- **Check:** Conditional logic handles None values properly
- **Check:** Dictionary keys match expected values

### Issue: Loss computation errors
- **Check:** Loss terms for disabled dimensions are set to 0.0
- **Check:** No operations on None tensors

### Issue: Poor reconstruction quality
- **Try:** Adjust KL balancing weights
- **Try:** Increase training epochs
- **Check:** Fixed scale value is appropriate (for minimal)

---

## Future Extensions

1. **Adaptive dimension selection:** Automatically determine which dimensions are needed
2. **Dimension-wise dropout:** Randomly disable dimensions during training for robustness
3. **Progressive training:** Start with minimal, gradually add dimensions
4. **Dataset-specific configs:** Auto-configure based on dataset characteristics
