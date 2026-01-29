"""Setup script for building C++ MCTS extension with pybind11."""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np

ext_modules = [
    Pybind11Extension(
        "alphazero.mcts.cpp.mcts_cpp",
        ["src/bindings.cpp"],
        include_dirs=[
            "src",
            np.get_include(),
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        cxx_std=17,
        extra_compile_args=["/O2", "/fp:fast"] if __import__("sys").platform == "win32" else ["-O3", "-ffast-math"],
    ),
]

setup(
    name="alphazero-mcts-cpp",
    packages=[],  # Disable auto-discovery
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
