#!/bin/bash
# Build the backend parity oracle (tests/oracle_train.cu) for this machine:
# macOS -> CPU backend (clang++), Linux -> CUDA (nvcc), both fp32.
# Output: ./build/oracle_train_ENV. Run ./build.sh ENV --cpu (macOS) or ./build.sh ENV
# (Linux) once first so raylib is downloaded.
#   ./tests/oracle_train.sh [ENV]   (default breakout)
set -e
ENV=${1:-breakout}
UPPER=$(echo "$ENV" | tr a-z A-Z)
COMMON=(-I. -Isrc -Iocean/$ENV -Ivendor
    -DENV_HEADER=\"ocean/$ENV/$ENV.h\" -DENV_NAME=$ENV -DPUFFER_ENV_NAME=\"$ENV\"
    -DPUFFER_$UPPER -DPRECISION_FLOAT)
mkdir -p build
if [ "$(uname -s)" = "Linux" ]; then
    CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    NCCL_I=""; NCCL_L=""
    if [ -n "${NCCL_HOME:-}" ]; then
        NCCL_I="-I$NCCL_HOME/include"
        NCCL_L="-L$NCCL_HOME/lib -Xlinker -rpath -Xlinker $NCCL_HOME/lib"
    fi
    $CUDA_HOME/bin/nvcc -O2 -arch=${NVCC_ARCH:-native} -std=c++17 "${COMMON[@]}" \
        -Iraylib-5.5_linux_amd64/include -I$CUDA_HOME/include -I$CUDA_HOME/include/cccl \
        $NCCL_I -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp \
        -Xcompiler=-Wno-narrowing --diag-suppress=2361,111,128 \
        tests/oracle_train.cu raylib-5.5_linux_amd64/lib/libraylib.a \
        -L$CUDA_HOME/lib64 $NCCL_L -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver \
        -lcurand -lm -Xlinker=-lpthread ${OMP_LIB:--lgomp} -lGL \
        -o build/oracle_train_$ENV
else
    OMP_PREFIX="$(brew --prefix libomp)"
    # METAL=1: CPU trainer + Metal GPU backend (binary oracle_train_ENV_metal)
    SUFFIX=""; METAL_ARGS=()
    if [ "${METAL:-0}" = "1" ]; then
        SUFFIX="_metal"
        METAL_ARGS=(-DPUF_TRAIN_METAL -x objective-c++ -fobjc-arc src/puf_metal.mm
            -framework Metal -framework MetalPerformanceShaders -framework Foundation)
    fi
    ${CXX:-clang++} -std=c++17 -O2 -g -w -Wno-narrowing "${COMMON[@]}" \
        -DPUF_TRAIN_CPU -fveclib=Darwin_libsystem_m -Iraylib-5.5_macos/include -DPLATFORM_DESKTOP \
        -Xclang -fopenmp -I"$OMP_PREFIX/include" \
        -x c++ tests/oracle_train.cu "${METAL_ARGS[@]}" -x none raylib-5.5_macos/lib/libraylib.a \
        -L"$OMP_PREFIX/lib" -lomp -framework Accelerate -framework Cocoa \
        -framework IOKit -framework CoreVideo -framework OpenGL \
        -o build/oracle_train_$ENV$SUFFIX
fi
echo "Built: ./build/oracle_train_$ENV${SUFFIX:-}"
