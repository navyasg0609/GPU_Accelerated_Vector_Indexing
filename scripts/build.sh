#!/usr/bin/env bash
# Builds everything into build/. Uses CMake when available, otherwise plain nvcc.
#   ARCH=sm_75 scripts/build.sh      # override the GPU architecture for the nvcc path
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v cmake >/dev/null 2>&1; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j "$(nproc)"
  exit 0
fi

if [[ -z "${ARCH:-}" ]]; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  ARCH="sm_${cap:-75}"
fi
NVCC_FLAGS=(-O3 -std=c++17 -arch="$ARCH" -lineinfo -Iinclude -Xcompiler -fopenmp,-O3,-march=native)
mkdir -p build/obj
objs=()
for src in src/*.cpp src/*.cu; do
  obj="build/obj/$(basename "${src%.*}").o"
  nvcc "${NVCC_FLAGS[@]}" -c "$src" -o "$obj"
  objs+=("$obj")
done
for app in apps/*.cu tests/test_correctness.cu; do
  nvcc "${NVCC_FLAGS[@]}" "$app" "${objs[@]}" -lcublas -lgomp -o "build/$(basename "${app%.cu}")"
done
echo "built: $(ls build | grep -v obj | tr '\n' ' ')"
