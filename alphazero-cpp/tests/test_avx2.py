#!/usr/bin/env python3
"""Test if AVX2 is enabled in the compiled C++ module."""

import sys
from pathlib import Path

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "build" / "Release"))

# Create a small C++ test snippet
test_cpp = """
#include <pybind11/pybind11.h>

namespace py = pybind11;

bool is_avx2_enabled() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

PYBIND11_MODULE(test_avx2_module, m) {
    m.def("is_avx2_enabled", &is_avx2_enabled);
}
"""

# For now, just check the compilation environment
print("Checking AVX2 compilation status...")
print()

# Try to detect from build logs or compiler flags
cmake_cache = Path(__file__).parent.parent / "build" / "CMakeCache.txt"
if cmake_cache.exists():
    with open(cmake_cache, 'r') as f:
        cache_content = f.read()
        if '/arch:AVX2' in cache_content or '-mavx2' in cache_content:
            print("[YES] AVX2 flag found in CMake cache")
        else:
            print("[NO] AVX2 flag NOT found in CMake cache")
            print()
            print("Current compiler flags:")
            for line in cache_content.split('\n'):
                if 'CMAKE_CXX_FLAGS' in line and not line.startswith('//'):
                    print(f"  {line}")
else:
    print("CMakeCache.txt not found")

print()
print("To enable AVX2, add to CMakeLists.txt:")
print("  if(MSVC)")
print("      target_compile_options(training PRIVATE /arch:AVX2)")
print("  else()")
print("      target_compile_options(training PRIVATE -mavx2)")
print("  endif()")
