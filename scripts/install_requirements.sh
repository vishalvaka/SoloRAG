#!/usr/bin/env bash
# --------------------------------------------------
# Install SoloRAG requirements with proper FAISS setup
#
# This script detects GPU availability and installs the appropriate
# FAISS package (faiss-cpu or faiss-gpu) along with compatible PyTorch.
#
# Usage:
#   ./scripts/install_requirements.sh          # Auto-detect hardware
#   ./scripts/install_requirements.sh --cpu    # Force CPU-only setup
#   ./scripts/install_requirements.sh --gpu    # Force GPU setup
# --------------------------------------------------

set -euo pipefail

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
FORCE_MODE=""
for arg in "$@"; do
    case $arg in
        --cpu)
            FORCE_MODE="cpu"
            ;;
        --gpu)
            FORCE_MODE="gpu"
            ;;
        *)
            echo -e "${RED}❌ Unknown argument: $arg${NC}"
            echo "Usage: $0 [--cpu|--gpu]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🔍 SoloRAG Requirements Installer${NC}"
echo "=================================="

# Function to detect GPU availability
detect_gpu() {
    local has_nvidia_smi=false
    local has_cuda_devices=false
    
    # Check if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        has_nvidia_smi=true
        echo -e "${GREEN}✅ nvidia-smi found${NC}"
        
        # Check if there are CUDA devices
        if nvidia-smi -L | grep -q "GPU"; then
            has_cuda_devices=true
            echo -e "${GREEN}✅ CUDA devices detected:${NC}"
            nvidia-smi -L | sed 's/^/   /'
        else
            echo -e "${YELLOW}⚠️  nvidia-smi found but no CUDA devices detected${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  nvidia-smi not found${NC}"
    fi
    
    # Return true if both conditions are met
    if [ "$has_nvidia_smi" = true ] && [ "$has_cuda_devices" = true ]; then
        return 0
    else
        return 1
    fi
}

# Determine which mode to use
if [ -n "$FORCE_MODE" ]; then
    MODE="$FORCE_MODE"
    echo -e "${BLUE}🎯 Mode: $MODE (forced)${NC}"
else
    if detect_gpu; then
        MODE="gpu"
        echo -e "${GREEN}🎮 Auto-detected: GPU mode${NC}"
    else
        MODE="cpu"
        echo -e "${YELLOW}💻 Auto-detected: CPU mode${NC}"
    fi
fi

echo ""

# Check if virtual environment is active
if [[ "${VIRTUAL_ENV:-}" == "" ]]; then
    echo -e "${YELLOW}⚠️  No virtual environment detected${NC}"
    echo "It's recommended to install in a virtual environment:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

echo -e "${BLUE}📦 Installing dependencies...${NC}"

# Install common requirements first
echo "Installing common requirements..."
pip install -r requirements/common.txt

# Install hardware-specific requirements
echo "Installing $MODE-specific requirements..."
if [ "$MODE" = "gpu" ]; then
    # Uninstall faiss-cpu to avoid conflicts with faiss-gpu
    echo "  Removing faiss-cpu to avoid conflicts..."
    pip uninstall faiss-cpu -y &>/dev/null || true
fi
pip install -r "requirements/$MODE.txt"

# Install main requirements (sentence-transformers, etc.)
echo "Installing main requirements..."
pip install -r requirements.txt

# Install dev requirements
echo "Installing development requirements..."
pip install -r requirements-dev.txt

echo ""
echo -e "${GREEN}✅ Installation completed!${NC}"

# Verify installation
echo -e "${BLUE}🔍 Verifying installation...${NC}"

python -c "
import sys
try:
    import faiss
    import torch
    import sentence_transformers
    
    print('✅ Core packages installed successfully')
    print(f'   📦 FAISS version: {faiss.__version__}')
    print(f'   📦 PyTorch version: {torch.__version__}')
    print(f'   📦 SentenceTransformers version: {sentence_transformers.__version__}')
    
    # Check GPU availability
    if '$MODE' == 'gpu':
        gpu_count = faiss.get_num_gpus()
        cuda_available = torch.cuda.is_available()
        has_gpu_functions = hasattr(faiss, 'StandardGpuResources')
        
        print(f'   🎮 FAISS GPUs detected: {gpu_count}')
        print(f'   🔥 PyTorch CUDA available: {cuda_available}')
        print(f'   ⚡ FAISS GPU functions: {has_gpu_functions}')
        
        if gpu_count > 0 and cuda_available and has_gpu_functions:
            print('🎉 GPU acceleration ready!')
        else:
            print('⚠️  GPU installation incomplete - some features may not work')
            if gpu_count == 0:
                print('   - FAISS cannot detect GPUs')
            if not cuda_available:
                print('   - PyTorch CUDA not available')
            if not has_gpu_functions:
                print('   - FAISS GPU functions not available')
    else:
        print('💻 CPU-only installation - GPU acceleration disabled')
        
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Verification error: {e}')
    sys.exit(1)
"

echo ""
echo -e "${GREEN}🎯 Installation Summary:${NC}"
echo "  Mode: $MODE"
echo "  FAISS: faiss-$MODE"
if [ "$MODE" = "gpu" ]; then
    echo "  PyTorch: CUDA-enabled"
    echo ""
    echo -e "${BLUE}💡 Next steps for GPU:${NC}"
    echo "  1. Test GPU functionality: ./scripts/run_tests.sh --gpu"
    echo "  2. Build GPU index: BUILD_MODE=gpu python scripts/build_index.py"
    echo "  3. Run with GPU Docker: docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml up"
else
    echo "  PyTorch: CPU-only"
    echo ""
    echo -e "${BLUE}💡 Next steps for CPU:${NC}"
    echo "  1. Run tests: ./scripts/run_tests.sh --no-gpu"
    echo "  2. Build index: python scripts/build_index.py"
    echo "  3. Run with CPU Docker: docker compose -f docker/compose.yaml up"
fi

echo ""
echo -e "${GREEN}🚀 Ready to go! Check the README for usage instructions.${NC}" 