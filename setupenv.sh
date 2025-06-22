#!/bin/bash

# setupenv.sh: Set up and activate the chromatic-orchestrator environment (source this script!)
# Logs all output to setupenv.log

# Guard: must be sourced, not executed
(return 0 2>/dev/null) || { echo "Please source this script: source setupenv.sh"; exit 1; }

DEBUG=1  # Set to 1 for debugging 0 for not

if [[ $DEBUG -eq 0 ]]; then
    set -e
fi

LOGFILE="setupenv.log"
exec > >(tee -a "$LOGFILE") 2>&1

ENV_NAME="chromatic-orchestrator"
PYTHON_VERSION="3.12"
VLLM_VERSION="0.9.1"

echo "=== Chromatic Orchestrator environment setup started at $(date) ==="

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    return 1 2>/dev/null || exit 1
fi

# Check if environment exists
if conda info --envs | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
    read -p "Do you want to delete and recreate it? [y/N]: " yn
    case "$yn" in
        [Yy]* )
            echo "Removing environment '$ENV_NAME'..."
            conda deactivate 2>/dev/null || true
            conda remove -n "$ENV_NAME" --all -y
            echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
            conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
            ;;
        * )
            echo "Keeping existing environment."
            ;;
    esac
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Print the current environment
CUR_ENV=$(conda info --json | python -c "import sys, json; print(json.load(sys.stdin)['active_prefix_name'])")
echo "Current active conda environment: $CUR_ENV"

if [[ "$CUR_ENV" != "$ENV_NAME" ]]; then
    echo "WARNING: Expected environment '$ENV_NAME' but current is '$CUR_ENV'."
    echo "Conda activation may not have worked as expected. You may need to manually conda activate $ENV_NAME."
    return 1 2>/dev/null || exit 1
fi

echo "=== Installing all packages via pip ==="

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install vLLM first (it will handle PyTorch dependencies)
echo "Installing vllm==$VLLM_VERSION (this will install PyTorch dependencies)..."
pip install "vllm==$VLLM_VERSION"

# Install core scientific packages
echo "Installing scientific computing packages..."
pip install numpy pandas matplotlib scikit-learn

# Install computer vision and image processing packages
echo "Installing computer vision packages..."
pip install opencv-python Pillow

# Install utility packages
echo "Installing utility packages..."
pip install tqdm prefetch-generator

# Install testing framework
echo "Installing testing packages..."
pip install pytest

# Install additional useful packages for development
echo "Installing development packages..."
pip install ipython jupyter

echo "=== Verifying installation ==="
python -c "
import sys
print(f'Python version: {sys.version}')

packages = [
    'torch', 'torchvision', 'cv2', 'matplotlib', 'numpy', 'pandas',
    'sklearn', 'PIL', 'tqdm', 'prefetch_generator', 'vllm'
]

for pkg in packages:
    try:
        if pkg == 'cv2':
            import cv2
            print(f'✓ OpenCV version: {cv2.__version__}')
        elif pkg == 'PIL':
            import PIL
            print(f'✓ Pillow version: {PIL.__version__}')
        elif pkg == 'sklearn':
            import sklearn
            print(f'✓ scikit-learn version: {sklearn.__version__}')
        else:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f'✓ {pkg} version: {version}')
    except ImportError as e:
        print(f'✗ Failed to import {pkg}: {e}')
"

echo "=== Testing CUDA availability ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available - will run on CPU')
"

echo "Environment setup complete. You are now in the '$ENV_NAME' environment."

# Optional: Run tests if they exist
if [[ -f "tests/test_torch.py" ]]; then
    echo "=== Running test suite: tests/test_torch.py ==="
    pytest -s tests/test_torch.py
fi

echo "=== Chromatic Orchestrator environment setup complete at $(date) ==="
echo ""
echo "Your environment includes:"
echo "  • PyTorch with CUDA support (installed via vLLM)"
echo "  • vLLM $VLLM_VERSION for inference"
echo "  • OpenCV for computer vision operations"
echo "  • scikit-learn for clustering algorithms"
echo "  • All packages needed for chromatic-orchestrator"
echo ""
echo "To activate this environment in the future:"
echo "  conda activate $ENV_NAME"