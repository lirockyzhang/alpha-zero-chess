#!/usr/bin/env bash
# setup_linux.sh — One-command setup for AlphaZero Chess on Linux
#
# Prerequisites: PyTorch already installed (with your desired CUDA version)
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/lirockyzhang/alpha-zero-chess/main/setup_linux.sh | bash
#   # or:
#   git clone https://github.com/lirockyzhang/alpha-zero-chess.git && cd alpha-zero-chess && ./setup_linux.sh

set -euo pipefail

# ─── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       AlphaZero Chess — Linux Setup                 ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo

# ─── Step 1: Install uv (if not present) ──────────────────────────────────────
info "Step 1/6: Checking for uv package manager..."

if command -v uv &>/dev/null; then
    ok "uv already installed: $(uv --version)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed: $(uv --version)"
    else
        fail "uv installation failed. Please install manually: https://docs.astral.sh/uv/"
    fi
fi
echo

# ─── Step 2: System dependencies ──────────────────────────────────────────────
info "Step 2/6: Checking system dependencies..."

MISSING_PKGS=()

check_dep() {
    local cmd=$1
    local pkg=$2
    if ! command -v "$cmd" &>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
}

check_dep cmake cmake
check_dep g++ build-essential
check_dep git git

# OpenMP: check for the header (glob doesn't expand inside [ -f ])
OMP_FOUND=false
[ -f /usr/include/omp.h ] && OMP_FOUND=true
compgen -G '/usr/lib/gcc/*/include/omp.h' &>/dev/null && OMP_FOUND=true
compgen -G '/usr/lib/gcc/*/*/include/omp.h' &>/dev/null && OMP_FOUND=true
if [ "$OMP_FOUND" = false ]; then
    MISSING_PKGS+=("libomp-dev")
fi

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    info "Missing packages: ${MISSING_PKGS[*]}"

    if command -v apt-get &>/dev/null; then
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update -qq && apt-get install -y -qq "${MISSING_PKGS[@]}"
        elif command -v sudo &>/dev/null; then
            info "Running: sudo apt-get install ${MISSING_PKGS[*]}"
            sudo apt-get update -qq && sudo apt-get install -y -qq "${MISSING_PKGS[@]}"
        else
            fail "Missing packages (${MISSING_PKGS[*]}) and no sudo available.\n  Please install them manually: apt-get install ${MISSING_PKGS[*]}"
        fi
        ok "System dependencies installed"
    else
        fail "Missing packages (${MISSING_PKGS[*]}) and apt-get not found.\n  Please install them with your system's package manager."
    fi
else
    ok "All system dependencies present"
fi
echo

# ─── Step 3: Clone repo + submodules (skip if already inside the repo) ────────
info "Step 3/6: Setting up repository..."

if [ -f "pyproject.toml" ] && grep -q "alpha-zero-chess" pyproject.toml 2>/dev/null; then
    ok "Already inside alpha-zero-chess repo"
    REPO_DIR="$(pwd)"
elif [ -d "alpha-zero-chess" ]; then
    ok "Repository already cloned"
    cd alpha-zero-chess
    REPO_DIR="$(pwd)"
else
    info "Cloning repository..."
    git clone https://github.com/lirockyzhang/alpha-zero-chess.git
    cd alpha-zero-chess
    REPO_DIR="$(pwd)"
    ok "Repository cloned"
fi

info "Initializing submodules..."
git submodule update --init --recursive

# Verify chess-library submodule
if [ -f "alphazero-cpp/third_party/chess-library/include/chess.hpp" ]; then
    ok "chess-library submodule present"
else
    fail "chess-library submodule is missing after init. Check git submodule output above."
fi
echo

# ─── Step 4: Python environment ───────────────────────────────────────────────
info "Step 4/6: Setting up Python environment..."

# Create venv using the system Python (which has torch pre-installed).
# uv venv doesn't have --system-site-packages, so we patch pyvenv.cfg after.
if [ ! -d ".venv" ]; then
    # Use system Python (>= 3.12) rather than a specific version,
    # so the venv matches the Python where torch is installed.
    SYS_PY=$(python3 --version 2>/dev/null | grep -oP '\d+\.\d+' || echo "3.12")
    info "Creating virtual environment (Python $SYS_PY)..."
    uv venv --python "$SYS_PY" --seed
else
    ok "Virtual environment already exists"
fi

# Enable system-site-packages so the venv inherits pre-installed torch
PYVENV_CFG=".venv/pyvenv.cfg"
if [ -f "$PYVENV_CFG" ]; then
    if grep -q "include-system-site-packages" "$PYVENV_CFG"; then
        sed -i 's/include-system-site-packages.*/include-system-site-packages = true/' "$PYVENV_CFG"
    else
        echo "include-system-site-packages = true" >> "$PYVENV_CFG"
    fi
    info "Enabled system-site-packages (inherit pre-installed PyTorch)"
fi

# Install all dependencies (torch/torchvision are optional — use pre-installed)
info "Installing Python dependencies..."
uv sync

ok "Python dependencies installed"
echo

# ─── Step 5: Build C++ module ─────────────────────────────────────────────────
info "Step 5/6: Building C++ MCTS engine..."

# Get pybind11 cmake directory from the venv
PYBIND11_DIR=$(uv run python -c "import pybind11; print(pybind11.get_cmake_dir())")
info "pybind11 cmake dir: $PYBIND11_DIR"

BUILD_DIR="alphazero-cpp/build"
mkdir -p "$BUILD_DIR"

info "Configuring with CMake..."
cmake -S alphazero-cpp -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    "-Dpybind11_DIR=$PYBIND11_DIR"

NPROC=$(nproc 2>/dev/null || echo 4)
info "Building with $NPROC cores..."
cmake --build "$BUILD_DIR" --config Release "-j$NPROC"

# On Linux, CMake puts .so in build/ (not build/Release/).
# train.py expects build/Release/, so create symlink.
RELEASE_DIR="$BUILD_DIR/Release"
mkdir -p "$RELEASE_DIR"

SO_FILE=$(find "$BUILD_DIR" -maxdepth 1 -name 'alphazero_cpp*.so' -print -quit)
if [ -n "$SO_FILE" ]; then
    SO_NAME=$(basename "$SO_FILE")
    TARGET="$RELEASE_DIR/$SO_NAME"
    if [ ! -e "$TARGET" ]; then
        ln -sf "../$SO_NAME" "$TARGET"
        info "Symlinked $SO_NAME -> Release/"
    fi
    ok "C++ module built: $SO_FILE"
else
    fail "Build completed but no alphazero_cpp*.so found in $BUILD_DIR"
fi
echo

# ─── Step 6: Verify ───────────────────────────────────────────────────────────
info "Step 6/6: Verifying installation..."

ERRORS=0

# Check C++ module loads
if uv run python -c "import alphazero_cpp; print('  alphazero_cpp:', dir(alphazero_cpp)[:5], '...')" 2>/dev/null; then
    ok "alphazero_cpp module loads"
else
    warn "alphazero_cpp module failed to load (sys.path may need build/Release/)"
    ERRORS=$((ERRORS + 1))
fi

# Check PyTorch
if uv run python -c "
import torch
print(f'  PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}', end='')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print()
" 2>/dev/null; then
    ok "PyTorch accessible"
else
    warn "PyTorch not found. Install it into the venv or system Python, e.g.:"
    warn "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    ERRORS=$((ERRORS + 1))
fi

# Check python-chess
if uv run python -c "import chess; print(f'  python-chess {chess.__version__}')" 2>/dev/null; then
    ok "python-chess accessible"
else
    warn "python-chess not found"
    ERRORS=$((ERRORS + 1))
fi

echo
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  Setup complete!${NC}"
else
    echo -e "${YELLOW}${BOLD}  Setup complete with $ERRORS warning(s).${NC}"
fi
echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
echo
echo "  Start training:"
echo "    cd $REPO_DIR"
echo "    uv run python alphazero-cpp/scripts/train.py \\"
echo "      --filters 256 --blocks 20 --iterations 100 \\"
echo "      --games-per-iter 50 --simulations 200 --workers 64"
echo
echo "  Or resume a previous run:"
echo "    uv run python alphazero-cpp/scripts/train.py --resume checkpoints/<run_dir>"
echo
