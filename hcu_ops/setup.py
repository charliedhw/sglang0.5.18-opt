import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx938")
here = os.path.dirname(os.path.abspath(__file__))
sources = sorted(glob.glob(os.path.join("csrc", "*.cu"))) + ["csrc/bind.cpp"]

setup(
    name="hcu_ops",
    version="0.1.0",
    packages=["hcu_ops"],
    package_data={"hcu_ops": ["skinny_cfg.json"]},
    ext_modules=[
        CUDAExtension(
            name="hcu_ops._C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--offload-arch=gfx938", "-save-temps=obj"]
                if os.environ.get("HCU_OPS_SAVE_TEMPS") else
                ["-O3", "--offload-arch=gfx938"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
