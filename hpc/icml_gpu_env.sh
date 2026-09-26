# CUDA environment for GPU jobs. Sourced automatically by icml_submit.sh whenever
# MSD_HPC_GRES requests a device. Do not run it on a login node.
#
# WHY THIS FILE EXISTS, AND WHY THE VERSION IS PINNED
# ---------------------------------------------------
# Two separate failures, both of which look like a working GPU:
#
# 1. MISSING libnvrtc. cupy-cuda12x needs libnvrtc.so.12 at run time and this cluster does
#    not have it on the default library path. Without it `cupy` imports fine and every probe
#    of the device fails. On 26 September job 669836 died on exactly this:
#      "CuPy failed to load libnvrtc.so.12: cannot open shared object file"
#    Worse, on 25 September job 669334 SURVIVED it and produced wrong output: the first cupy
#    probe failed, then the same process imported torch and touched CUDA -- which loads
#    torch's OWN bundled nvrtc into the process -- and the second probe then succeeded. The
#    result was a CPU-only cost table beside a GPU-inclusive scaling table from one process.
#    So a missing library here does not fail cleanly; it fails *selectively*, depending on
#    whether torch happens to have been imported first.
#
# 2. WRONG CUDA VERSION. The system default is CUDA 13.3. cupy-cuda12x JIT-compiles its
#    reduction kernels against whatever CUDA_PATH points at, and against 13.3 headers that
#    compile fails inside cuda_fp4.hpp. cuBLAS calls need no JIT and keep working, so matrix
#    products succeed while every .max()/.sum() reduction in the BP recursion dies. Pinning
#    to the 12.4 module gives headers and libraries from one release.
#    (This diagnosis is inherited from the earlier deployment's gpu_env.sh, which recorded it
#    after the first cluster attempt; see also hpc/bocconi_gpu.sbatch on the cupy < 14 pin.)
#
# The lesson both failures share: verify the device with markov_score.backend.gpu_available()
# and check `cupy_probe_error` in the output's params file. A GPU job that quietly runs on the
# CPU is worse than one that dies.
export CUDA_PATH=/software/cuda/12.4
export LD_LIBRARY_PATH=/software/cuda/12.4/lib64:${LD_LIBRARY_PATH:-}
