#!/bin/bash
# Build llama.cpp with CUDA support for TurboQuant

set -e

echo "============================================================"
echo "llama.cpp CUDA Build Script for TurboQuant"
echo "============================================================"

# Configuration
LLAMA_CPP_URL="https://github.com/TheTom/turboquant_plus.git"
BUILD_DIR="$(pwd)/llama.cpp/build"
INSTALL_DIR="$(pwd)/llama.cpp"
CUDA_ARCH=${CUDA_ARCH:-"70;75;80;86;89"}
USE_CUDA=${USE_CUDA:-ON}
USE_METAL=${USE_METAL:-OFF}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting llama.cpp CUDA build...${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | cut -d' ' -f6 | cut -d',' -f1)
    echo -e "${GREEN}✓ CUDA found: $CUDA_VERSION${NC}"
else
    echo -e "${RED}✗ CUDA not found${NC}"
    echo "Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check for CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓ CMake found: $CMAKE_VERSION${NC}"
else
    echo -e "${RED}✗ CMake not found${NC}"
    echo "Install with: sudo apt-get install cmake"
    exit 1
fi

# Check for git
if command -v git &> /dev/null; then
    echo -e "${GREEN}✓ Git found${NC}"
else
    echo -e "${RED}✗ Git not found${NC}"
    exit 1
fi

echo ""

# Clone or update llama.cpp
if [ -d "$BUILD_DIR/../.git" ]; then
    echo -e "${YELLOW}Updating existing llama.cpp repository...${NC}"
    cd "$BUILD_DIR/.."
    git pull
else
    echo -e "${YELLOW}Cloning llama.cpp repository...${NC}"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git clone "$LLAMA_CPP_URL" .
fi

echo ""

# Create build directory
echo -e "${YELLOW}Creating build directory...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
echo -e "${YELLOW}Configuring CMake with CUDA support...${NC}"
echo "  CUDA: $USE_CUDA"
echo "  Metal: $USE_METAL"
echo "  CUDA Architectures: $CUDA_ARCH"
echo ""

cmake .. \
    -DGGML_CUDA=$USE_CUDA \
    -DGGML_METAL=$USE_METAL \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release

echo ""

# Build
echo -e "${YELLOW}Building llama.cpp (this may take 10-20 minutes)...${NC}"
cmake --build . --config Release -j$(nproc)

echo ""

# Verify build
echo -e "${YELLOW}Verifying build...${NC}"
if [ -f "bin/main" ] && [ -f "bin/server" ] && [ -f "bin/quantize" ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo "Binaries created:"
    ls -lh bin/
else
    echo -e "${RED}✗ Build failed - some binaries missing${NC}"
    exit 1
fi

echo ""

# Copy binaries to install directory
echo -e "${YELLOW}Installing binaries to $INSTALL_DIR...${NC}"
cp bin/* "$INSTALL_DIR/"

echo ""

# Test installation
echo -e "${YELLOW}Testing installation...${NC}"
if [ -f "$INSTALL_DIR/main" ]; then
    VERSION=$("$INSTALL_DIR/main" --version 2>&1 | head -n1)
    echo -e "${GREEN}✓ llama.cpp main binary working${NC}"
    echo "  Version: $VERSION"
else
    echo -e "${RED}✗ Binary test failed${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo -e "${GREEN}Build Complete!${NC}"
echo "============================================================"
echo ""
echo "Binaries installed to: $INSTALL_DIR"
echo ""
echo "Quick test:"
echo "  cd $INSTALL_DIR"
echo "  ./main -m your-model.gguf -p \"Hello\" -n 32"
echo ""
echo "With TurboQuant KV cache:"
echo "  ./main -m model.gguf -p \"test\" -n 128 --gpu-layers 32 \\"
echo "         --kv-cache-type-k q8_0 --kv-cache-type-v turbo4"
echo ""
echo "Python integration:"
echo "  python test_cuda_integration.py"
echo ""
echo "============================================================"
