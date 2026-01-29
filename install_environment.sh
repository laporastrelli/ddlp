#!/bin/bash
# Installation script for DDLP with multiple fallback strategies

echo "================================================"
echo "DDLP Environment Installation Script"
echo "================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Function to check if environment exists
env_exists() {
    conda env list | grep -q "^$1 "
}

ENV_NAME="ddlp"

echo "Checking if environment '$ENV_NAME' already exists..."
if env_exists "$ENV_NAME"; then
    echo "WARNING: Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and start fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Aborting installation."
        exit 1
    fi
fi

echo ""
echo "================================================"
echo "Fast Installation: Python + pip only (Recommended)"
echo "================================================"
echo "Creating minimal conda environment with Python 3.8..."

# Create basic environment with just Python and pip - no extra packages
conda create -n $ENV_NAME python=3.8 pip -y

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create basic conda environment."
    exit 1
fi

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ""
echo "Installing PyTorch with CUDA 11.8..."
# Install PyTorch first (adjust cuda version if needed)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

if [ $? -ne 0 ]; then
    echo "WARNING: PyTorch installation failed. Trying CPU version..."
    pip install torch==2.0.1 torchvision==0.15.2
fi

echo ""
echo "Installing scientific packages with pip (faster, fewer conflicts)..."
pip install numpy==1.24.3 matplotlib==3.7.1 scipy==1.8.1

echo ""
echo "Installing remaining packages..."
pip install scikit-image==0.19.2 h5py==3.1.0 imageio==2.28.1 tqdm==4.65.0

echo ""
echo "Installing DDLP-specific packages..."
pip install accelerate==0.19.0 einops==0.6.1 opencv-python piqa==1.3.1

echo ""
echo "Installing GUI packages (optional)..."
pip install ttkthemes==3.2.2 ttkwidgets==0.13.0

echo ""
echo "Installing notebook support (optional)..."
pip install notebook==6.5.4 ipykernel

echo ""
echo "Installing ffmpeg via conda (required for video processing)..."
conda install -n $ENV_NAME -c conda-forge ffmpeg=4.2.2 -y 2>/dev/null || echo "Note: ffmpeg installation optional, you can install it separately if needed"

echo ""
echo "================================================"
echo "Verifying installation..."
echo "================================================"

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import h5py; print(f'h5py: {h5py.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"

echo ""
echo "================================================"
echo "Installation complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "    conda activate $ENV_NAME"
echo ""
echo "To test if everything works, try:"
echo "    python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
