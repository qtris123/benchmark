#!/usr/bin/env python3
"""
sum_naive_pytorch.py — Naive PyTorch element-wise array addition at multiple sizes.

Designed for Nsight Systems profiling:
  - NVTX ranges mark each size block so the GUI shows them as colored bands
  - cudaProfilerStart / cudaProfilerStop bracket the hot section so
    nsys --capture-range=cudaProfilerApi captures only the profiled iterations
  - Each size runs WARMUP_ITERS un-profiled + PROFILE_ITERS profiled iterations
    to give nsys a strong, clean signal
"""

import torch
import torch.cuda.nvtx as nvtx
from torch.cuda import profiler as cuda_profiler

# ── Config ────────────────────────────────────────────────────────────────────
# Sizes chosen to span from 1 M to 128 M float32 elements (~4 MB to 512 MB/array)
# keeping total device memory well within A100 limits.
SIZES        = [1 << 20,   # 1 M   (4 MB)
                1 << 22,   # 4 M   (16 MB)
                1 << 24,   # 16 M  (64 MB)
                1 << 26,   # 64 M  (256 MB)
                1 << 27]   # 128 M (512 MB)
WARMUP_ITERS  = 20   # un-profiled, to drive JIT and caching
PROFILE_ITERS = 100  # iterations captured by nsys

DEVICE = "cuda"
DTYPE  = torch.float32

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    dev = torch.device(DEVICE)
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Sizes  : {[f'{N//1024//1024}M' for N in SIZES]} elements (float32)")
    print()

    # ── Allocate buffers once at max size, then slice views ───────────────────
    MAX_N = SIZES[-1]
    a_full = torch.rand(MAX_N, device=dev, dtype=DTYPE)
    b_full = torch.rand(MAX_N, device=dev, dtype=DTYPE)
    torch.cuda.synchronize()

    # ── Warm up CUDA runtime / driver before we start profiling ───────────────
    nvtx.range_push("WARMUP_GLOBAL")
    _ = torch.add(a_full[:SIZES[0]], b_full[:SIZES[0]])
    torch.cuda.synchronize()
    nvtx.range_pop()

    # ── Open profiler capture window ──────────────────────────────────────────
    cuda_profiler.start()   # nsys --capture-range=cudaProfilerApi starts here

    for N in SIZES:
        a = a_full[:N]
        b = b_full[:N]

        # -- un-profiled warmup for this size --
        nvtx.range_push(f"WARMUP_N{N // 1024 // 1024}M")
        for _ in range(WARMUP_ITERS):
            c = torch.add(a, b)
        torch.cuda.synchronize()
        nvtx.range_pop()

        # -- profiled iterations --
        label = f"PYTORCH_SUM_N{N // 1024 // 1024}M_iters{PROFILE_ITERS}"
        nvtx.range_push(label)
        for _ in range(PROFILE_ITERS):
            c = torch.add(a, b)
        torch.cuda.synchronize()
        nvtx.range_pop()

        print(f"  [naive]  N={N:>10,}  sum shape={c.shape}  sum[0]={c[0].item():.4f}")

    cuda_profiler.stop()    # nsys capture window ends here

    print()
    print("Done — naive PyTorch sum complete.")


if __name__ == "__main__":
    main()
