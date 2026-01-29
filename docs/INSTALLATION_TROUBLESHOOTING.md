# DDLP Installation Guide - Troubleshooting Dependency Conflicts

## Problem Summary

The original `environment.yml` file has overly specific package versions that cause dependency conflicts in modern conda environments. This is common with exported conda environments that include exact versions of system libraries.

## Why `conda env create -n ddlp` Fails

**Correction:** This command doesn't cause conflicts because it only creates an empty environment with no packages. The correct command you likely meant is:
```bash
conda env create -f environment.yml
```

## Root Causes of Conflicts

1. **Over-specified versions** - The original `environment.yml` has exact versions like `_libgcc_mutex=0.1` that may no longer be available
2. **CUDA library conflicts** - Very specific CUDA 11.8 library versions that conflict with system CUDA
3. **Python 3.8.16** - Old Python version with dependencies that may conflict with newer packages
4. **Channel mixing** - Packages from multiple channels (pytorch, nvidia, conda-forge, defaults) can conflict

---

## Recommended Solutions (in order of preference)

### Option 1: Use Simplified Environment File (EASIEST)

I've created `environment_simplified.yml` with flexible version constraints:

```bash
cd /data2/users/lr4617/ddlp
conda env create -f environment_simplified.yml
conda activate ddlp
```

This uses version ranges (e.g., `>=2.0.0,<2.1.0`) instead of exact versions, giving conda more flexibility to resolve dependencies.

---

### Option 2: Use Automated Installation Script (RECOMMENDED)

I've created `install_environment.sh` that tries multiple strategies:

```bash
cd /data2/users/lr4617/ddlp
bash install_environment.sh
```

The script will:
1. Try the simplified environment.yml first
2. If that fails, create a minimal environment and install packages step-by-step
3. Verify the installation
4. Show helpful error messages

---

### Option 3: Manual Installation (MOST RELIABLE)

If both above fail, install manually:

```bash
# 1. Create base environment with Python 3.8
conda create -n ddlp python=3.8 pip -y

# 2. Activate it
conda activate ddlp

# 3. Install PyTorch with CUDA (or CPU if no GPU)
# For CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch==2.0.1 torchvision==0.15.2

# 4. Install conda packages (fewer conflicts when installed after PyTorch)
conda install -c conda-forge numpy=1.24.3 matplotlib=3.7.1 scipy=1.8.1 \
    scikit-image=0.19.2 h5py=3.1.0 imageio=2.28.1 tqdm=4.65.0 \
    notebook=6.5.4 ffmpeg=4.2.2 -y

# 5. Install remaining packages with pip
pip install accelerate==0.19.0 einops==0.6.1 opencv-python==3.4.18.65 \
    piqa==1.3.1 ttkthemes==3.2.2 ttkwidgets==0.13.0

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

### Option 4: Use Existing Environment (IF AVAILABLE)

If you already have an environment with PyTorch 2.0.x:

```bash
conda activate diffusion-forcing  # or your existing env
pip install -r /data2/users/lr4617/ddlp/requirements.txt
```

This installs only the DDLP-specific packages into your existing environment.

---

## Verification

After installation, verify everything works:

```bash
conda activate ddlp
python -c "
import torch
import torchvision
import numpy as np
import h5py
import accelerate
import einops

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ Torchvision:', torchvision.__version__)
print('✓ NumPy:', np.__version__)
print('✓ h5py:', h5py.__version__)
print('✓ Accelerate:', accelerate.__version__)
print('✓ einops:', einops.__version__)
print('All packages installed successfully!')
"
```

---

## Common Issues & Solutions

### Issue: CUDA version mismatch
**Error:** `CUDA version X.X does not match PyTorch CUDA version Y.Y`

**Solution:** 
```bash
# Check your system CUDA version
nvidia-smi

# Install PyTorch matching your CUDA version
# For CUDA 11.7:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# For CUDA 12.1:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

### Issue: opencv-python conflicts
**Error:** `opencv-python==3.4.18.65` not found or conflicts

**Solution:**
```bash
# Use a more recent version
pip install opencv-python>=4.5.0
```

### Issue: h5py won't install
**Error:** HDF5 library errors

**Solution:**
```bash
# Install HDF5 system libraries first
conda install -c conda-forge hdf5=1.10.6 -y
pip install h5py==3.1.0
```

### Issue: ffmpeg not available
**Error:** `ffmpeg` command not found

**Solution:**
```bash
conda install -c conda-forge ffmpeg=4.2.2 -y
```

---

## Alternative: Use Docker (Advanced)

If conda continues to have issues, consider using Docker:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN pip install numpy==1.24.3 matplotlib==3.7.1 scipy==1.8.1 \
    scikit-image==0.19.2 h5py==3.1.0 imageio==2.28.1 tqdm==4.65.0 \
    accelerate==0.19.0 einops==0.6.1 opencv-python piqa==1.3.1
```

---

## Why the Original environment.yml Has Issues

The original file was likely created with:
```bash
conda env export > environment.yml
```

This captures EVERY package and dependency with exact versions, including:
- System libraries (`_libgcc_mutex`, `ld_impl_linux-64`)
- Build-specific packages
- Platform-specific dependencies

This makes it very fragile across different systems, conda versions, and time.

**Better practice:** Use version ranges or only specify direct dependencies.

---

## Need Help?

If all else fails:
1. Share the exact error message
2. Check your system: `nvidia-smi` (for CUDA version)
3. Check conda version: `conda --version`
4. Try creating a fresh conda installation or using mamba: `conda install -c conda-forge mamba -y`
