#!/bin/bash
set -euo pipefail
exec > /root/setup_log.txt 2>&1

echo "=== Step 1: Repo setup ==="
cd /root/alpha-zero-chess
git submodule update --init --recursive
echo "Submodules OK"

echo "=== Step 2: Install deps into existing venv ==="
source /venv/main/bin/activate
cd /root/alpha-zero-chess
uv pip install pybind11 python-chess tqdm numpy flask flask-cors flask-socketio matplotlib seaborn pandas scikit-learn scikit-image scipy
echo "Deps installed"

echo "=== Step 3: Build C++ ==="
PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "pybind11 dir: $PYBIND11_DIR"

BUILD_DIR=alphazero-cpp/build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cmake -S alphazero-cpp -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    "-Dpybind11_DIR=$PYBIND11_DIR"

NPROC=$(nproc)
echo "Building with $NPROC cores..."
cmake --build "$BUILD_DIR" --config Release "-j$NPROC"

# Symlink for train.py
RELEASE_DIR="$BUILD_DIR/Release"
mkdir -p "$RELEASE_DIR"
SO_FILE=$(find "$BUILD_DIR" -maxdepth 1 -name "alphazero_cpp*.so" -print -quit)
if [ -n "$SO_FILE" ]; then
    SO_NAME=$(basename "$SO_FILE")
    ln -sf "../$SO_NAME" "$RELEASE_DIR/$SO_NAME"
    echo "Symlinked: $SO_NAME -> Release/"
fi

echo "=== Step 4: Verify ==="
python -c "
import sys
sys.path.insert(0, 'alphazero-cpp/build')
sys.path.insert(0, 'alphazero-cpp/build/Release')
import alphazero_cpp
print('alphazero_cpp OK:', dir(alphazero_cpp)[:5])
"

python -c "
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()} GPU={gpu}')
"

echo "=== SETUP COMPLETE ==="
