#!/bin/bash
# Build and run the Metal-vs-CPU per-kernel differential test (macOS).
# Run ./build.sh breakout --cpu once first so raylib is downloaded.
set -e
OMP_PREFIX="$(brew --prefix libomp)"
mkdir -p build
${CXX:-clang++} -std=c++17 -O1 -g -w -Wno-narrowing \
    -DPUF_TRAIN_CPU -DPUF_TRAIN_METAL -DPRECISION_FLOAT \
    -I. -Isrc -Iocean/breakout -Ivendor -Iraylib-5.5_macos/include \
    -DENV_HEADER=\"ocean/breakout/breakout.h\" -DENV_NAME=breakout \
    -DPUFFER_ENV_NAME=\"breakout\" -DPLATFORM_DESKTOP -DPUFFER_BREAKOUT \
    -Xclang -fopenmp -I"$OMP_PREFIX/include" \
    -x c++ tests/test_metal_kernels.cpp -x objective-c++ -fobjc-arc src/puf_metal.mm \
    -x none raylib-5.5_macos/lib/libraylib.a \
    -L"$OMP_PREFIX/lib" -lomp -framework Accelerate -framework Metal \
    -framework MetalPerformanceShaders -framework Foundation -framework Cocoa \
    -framework IOKit -framework CoreVideo -framework OpenGL \
    -o build/test_metal_kernels
./build/test_metal_kernels
