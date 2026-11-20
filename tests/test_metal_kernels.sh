#!/bin/bash
# Build and run the Metal-vs-CPU per-kernel differential test (macOS).
#   ./tests/test_metal_kernels.sh [ENV] [--bf16]   (default breakout, fp32)
# Run ./build.sh breakout --cpu once first so raylib is downloaded.
set -e
ENV=breakout
PREC=-DPRECISION_FLOAT
for a in "$@"; do
    case $a in
        --bf16) PREC= ;;
        *) ENV=$a ;;
    esac
done
UPPER=$(echo "$ENV" | tr a-z A-Z)
OMP_PREFIX="$(brew --prefix libomp)"
mkdir -p build
${CXX:-clang++} -std=c++17 -O1 -g -w -Wno-narrowing \
    -DPUF_TRAIN_CPU -DPUF_TRAIN_METAL $PREC \
    -I. -Isrc -Iocean/$ENV -Ivendor -Iraylib-5.5_macos/include \
    -DENV_HEADER=\"ocean/$ENV/$ENV.h\" -DENV_NAME=$ENV \
    -DPUFFER_ENV_NAME=\"$ENV\" -DPLATFORM_DESKTOP -DPUFFER_$UPPER \
    -Xclang -fopenmp -I"$OMP_PREFIX/include" \
    -x c++ tests/test_metal_kernels.cpp -x objective-c++ -fobjc-arc src/puf_metal.mm \
    -x none raylib-5.5_macos/lib/libraylib.a \
    -L"$OMP_PREFIX/lib" -lomp -framework Accelerate -framework Metal \
    -framework MetalPerformanceShaders -framework Foundation -framework QuartzCore -framework Cocoa \
    -framework IOKit -framework CoreVideo -framework OpenGL \
    -o build/test_metal_kernels_$ENV${PREC:+_f32}
./build/test_metal_kernels_$ENV${PREC:+_f32}
