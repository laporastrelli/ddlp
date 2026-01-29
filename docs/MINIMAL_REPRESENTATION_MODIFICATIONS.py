"""
Code Modifications Required for Minimal and Intermediate Latent Representations
==================================================================================

This file documents the changes needed in the DDLP codebase to support:
1. INTERMEDIATE representation: z_p(2D) + z_s(2D) + z_f(3D) = 7D per particle
   - Removes: z_d (depth), z_t (transparency)
   
2. MINIMAL representation: z_p(2D) + z_f(3D) = 5D per particle
   - Removes: z_d (depth), z_t (transparency), z_s (scale)

IMPORTANT: Simply changing the config file is NOT sufficient. The architecture
has hard-coded neural network heads that must be modified.

================================================================================
REQUIRED MODIFICATIONS:
================================================================================

1. modules/modules.py - ParticleAttributeEncoder class
------------------------------------------------------
Location: Line ~703-850

CURRENT CODE has these heads:
    self.x_head = nn.Linear(hidden_dim_2, 2)
    self.y_head = nn.Linear(hidden_dim_2, 2)
    self.scale_xy_head = nn.Linear(hidden_dim_2, 4)
    self.obj_on_head = nn.Linear(hidden_dim_2, 2)      # <- REMOVE for both
    self.depth_head = nn.Linear(hidden_dim_2, 2)       # <- REMOVE for both

MODIFICATIONS:

A) For INTERMEDIATE representation:
   - Comment out or conditionally disable:
     * self.obj_on_head
     * self.depth_head
   - Keep scale_xy_head
   
B) For MINIMAL representation:
   - Comment out or conditionally disable:
     * self.obj_on_head
     * self.depth_head
     * self.scale_xy_head
   - Use fixed scale value from config

In the forward() method (~line 764-845):
   - Skip computing obj_on and depth outputs
   - For minimal: use fixed scale instead of learned scale


2. modules/dynamics_modules.py - ParticleFeatureProjection class
----------------------------------------------------------------
Location: Line ~254-315

CURRENT CODE:
    self.particle_dim = 2 + 2 + 1 + 1 + in_features_dim
    # [z, z_scale, z_obj_on, z_depth, z_features]

MODIFICATIONS:

A) For INTERMEDIATE:
    self.particle_dim = 2 + 2 + in_features_dim
    # [z, z_scale, z_features]
    
    In forward():
    - Remove z_obj_on_proj and z_depth_proj from concatenation
    - z_all = torch.cat([z_proj, z_scale_proj, z_features_proj], dim=-1)

B) For MINIMAL:
    self.particle_dim = 2 + in_features_dim
    # [z, z_features]
    
    In forward():
    - Remove z_obj_on_proj, z_depth_proj, and z_scale_proj
    - z_all = torch.cat([z_proj, z_features_proj], dim=-1)


3. modules/dynamics_modules.py - ParticleFeatureDecoder class
-------------------------------------------------------------
Location: Line ~317-410

CURRENT CODE has these heads:
    self.x_head = nn.Linear(hidden_dim, 2)
    self.y_head = nn.Linear(hidden_dim, 2)
    self.scale_xy_head = nn.Linear(hidden_dim, 4)
    self.obj_on_head = nn.Linear(hidden_dim, 2)        # <- REMOVE for both
    self.depth_head = nn.Linear(hidden_dim, 2)         # <- REMOVE for both
    self.features_head = nn.Linear(hidden_dim, 2 * features_dim)

MODIFICATIONS:

A) For INTERMEDIATE:
   - Comment out obj_on_head and depth_head
   - Keep scale_xy_head
   
B) For MINIMAL:
   - Comment out obj_on_head, depth_head, and scale_xy_head

In forward() method:
   - Skip computing corresponding outputs
   - Return None or zero tensors for removed dimensions


4. modules/dynamics_modules.py - DynamicsDLP class
--------------------------------------------------
Location: Line ~412-614

In __init__(), update the feature projection and decoder initialization
to use the appropriate particle dimensions based on config flags.

In forward(), handle missing dimensions appropriately when calling
encoder and decoder.


5. models.py - FgDLP, ObjectDLP, ObjectDynamicsDLP classes
-----------------------------------------------------------

Multiple locations where depth and transparency are used:

A) In encode_all() method:
   - Skip encoding obj_on and depth when flags are False
   - Return None or dummy values for disabled dimensions

B) In loss computation functions:
   - Skip KL divergence terms for disabled dimensions
   - Set corresponding loss terms to 0.0

Key locations to modify:
   - Line ~160-274: encode_all() in FgDLP
   - Line ~1500-1650: loss computation in ObjectDLP
   - Similar patterns in ObjectDynamicsDLP


6. Add configuration parameter handling
----------------------------------------

In train_ddlp.py and train_dlp.py:

Add logic to read new config flags:
    use_depth = config.get('use_depth', True)
    use_transparency = config.get('use_transparency', True)
    use_scale = config.get('use_scale', True)
    fixed_scale_value = config.get('fixed_scale_value', 0.25)

Pass these flags to model initialization.


================================================================================
IMPLEMENTATION APPROACH:
================================================================================

Option 1: CONDITIONAL LOGIC (Recommended)
------------------------------------------
Add flags to control which dimensions are active:
- Pros: Single codebase, easy to experiment
- Cons: More complex code with conditionals

Option 2: SEPARATE MODEL CLASSES
---------------------------------
Create new model classes (e.g., ObjectDLPMinimal, ObjectDLPIntermediate)
- Pros: Clean separation, easier to maintain
- Cons: Code duplication

Option 3: HYBRID
----------------
Use configuration flags with factory functions:
- Create models based on config
- Override specific methods as needed


================================================================================
TESTING CHECKLIST:
================================================================================

After implementing changes, verify:
□ Model initializes without errors
□ Forward pass works with new dimensions
□ Loss computation handles missing dimensions
□ Backward pass and gradient flow work correctly
□ Video prediction/generation works
□ Checkpoint saving/loading compatible
□ Evaluation metrics still compute correctly


================================================================================
EXPECTED BENEFITS:
================================================================================

INTERMEDIATE (7D):
- ~22% fewer parameters in attribute encoder heads
- Faster training (fewer dimensions to predict)
- Potentially better generalization (less overfitting)
- Still handles scale variation if needed

MINIMAL (5D):
- ~44% fewer parameters in attribute encoder heads
- Fastest training
- Most compact latent representation
- Best for simple datasets like Balls-Interactions


================================================================================
"""

# Example implementation snippet for conditional logic:

def create_particle_encoder(config):
    """Factory function to create encoder based on config."""
    use_depth = config.get('use_depth', True)
    use_transparency = config.get('use_transparency', True)
    use_scale = config.get('use_scale', True)
    
    if not use_depth and not use_transparency and not use_scale:
        return ParticleAttributeEncoderMinimal(...)
    elif not use_depth and not use_transparency:
        return ParticleAttributeEncoderIntermediate(...)
    else:
        return ParticleAttributeEncoder(...)
