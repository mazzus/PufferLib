#!/bin/bash
# Build the env-only SPS harness for one .h env: ./tests/bench_env_sps.sh ENV
# Output: ./build/bench_env_sps_ENV. Run ./build.sh ENV --cpu once first so the
# raylib (and any env-specific) dependencies are downloaded.
set -e
ENV=$1
[ -n "$ENV" ] || { echo "Usage: $0 ENV"; exit 1; }
[ -f "ocean/$ENV/$ENV.h" ] || { echo "Error: ocean/$ENV/$ENV.h not found"; exit 1; }
if [ "$(uname -s)" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    OMP=(-fopenmp)
    LD=(-lGL -ldl)
else
    RAYLIB_NAME='raylib-5.5_macos'
    OMP_PREFIX="$(brew --prefix libomp)"
    OMP=(-Xclang -fopenmp -I"$OMP_PREFIX/include" -L"$OMP_PREFIX/lib" -lomp)
    LD=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
fi
if [ "$(uname -m)" = "x86_64" ]; then SIMD=(-mavx2 -mfma); else SIMD=(); fi
UPPER=$(echo "$ENV" | tr a-z A-Z)
mkdir -p build
${CC:-clang} -O2 "${SIMD[@]}" -Wno-narrowing -Wno-unreachable-code \
    -Wno-incompatible-pointer-types-discards-qualifiers \
    -I. -Isrc -Iocean/$ENV -Ivendor -I$RAYLIB_NAME/include \
    -DPLATFORM_DESKTOP -DPUFFER_$UPPER \
    -DENV_HEADER=\"ocean/$ENV/$ENV.h\" -DPUFFER_ENV_NAME=\"$ENV\" \
    tests/bench_env_sps.c $RAYLIB_NAME/lib/libraylib.a \
    -lm -lpthread "${OMP[@]}" "${LD[@]}" \
    -o build/bench_env_sps_$ENV
echo "Built: ./build/bench_env_sps_$ENV"
