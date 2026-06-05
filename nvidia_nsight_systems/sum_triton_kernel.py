#!/usr/bin/env python3
"""
sum_triton_kernel.py — Optimized Triton kernel for element-wise array addition.

Kernel design:
  - BLOCK_SIZE = 512: each Triton program instance processes 512 contiguous
    float32 elements in parallel across threads in a warp-friendly tiling.
  - Grid = ceil(N / 512) program instances, fully parallelizing the workload.
  - Coalesced loads/stores: consecutive program IDs map to consecutive memory
    regions, ensuring optimal L2 / HBM bandwidth utilization.

Profiling setup mirrors sum_naive_pytorch.py (same SIZES / ITERS) so the two
nsys reports can be compared side-by-side in the Nsight Systems GUI.
"""

import torch
import torch.cuda.nvtx as nvtx
from torch.cuda import profiler as cuda_profiler
import triton
import triton.language as tl

# ── Triton kernel ─────────────────────────────────────────────────────────────
BLOCK_SIZE = 512

@triton.jit
def add_kernel(
    x_ptr,          # pointer to first  input vector
    y_ptr,          # pointer to second input vector
    out_ptr,        # pointer to output vector
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,   # tile size — must be a compile-time constant
):
    """Each program instance adds BLOCK_SIZE consecutive elements."""
    pid         = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets     = block_start + tl.arange(0, BLOCK_SIZE)
    mask        = offsets < n_elements          # guard against out-of-bounds

    x   = tl.load(x_ptr   + offsets, mask=mask)
    y   = tl.load(y_ptr   + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Launch the Triton add kernel and return result tensor."""
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    out         = torch.empty_like(x)
    n_elements  = x.numel()
    grid        = (triton.cdiv(n_elements, BLOCK_SIZE),)   # 1-D grid
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ── Config (must match sum_naive_pytorch.py for apples-to-apples comparison) ──
SIZES        = [1 << 20,   # 1 M   (4 MB)
                1 << 22,   # 4 M   (16 MB)
                1 << 24,   # 16 M  (64 MB)
                1 << 26,   # 64 M  (256 MB)
                1 << 27]   # 128 M (512 MB)
WARMUP_ITERS  = 20
PROFILE_ITERS = 100

DEVICE = "cuda"
DTYPE  = torch.float32


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    dev = torch.device(DEVICE)
    print(f"Device    : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch   : {torch.__version__}")
    print(f"Triton    : {triton.__version__}")
    print(f"BLOCK_SIZE: {BLOCK_SIZE}")
    print(f"Sizes     : {[f'{N//1024//1024}M' for N in SIZES]} elements (float32)")
    print()

    # ── Allocate buffers ──────────────────────────────────────────────────────
    MAX_N   = SIZES[-1]
    a_full  = torch.rand(MAX_N, device=dev, dtype=DTYPE)
    b_full  = torch.rand(MAX_N, device=dev, dtype=DTYPE)
    torch.cuda.synchronize()

    # ── Force Triton JIT compilation before any profiling ────────────────────
    # Run a small dummy call so the kernel is compiled and cached; subsequent
    # calls go directly to the cached cubin — no JIT overhead in the trace.
    nvtx.range_push("TRITON_JIT_COMPILE")
    _ = triton_add(a_full[:SIZES[0]], b_full[:SIZES[0]])
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
            c = triton_add(a, b)
        torch.cuda.synchronize()
        nvtx.range_pop()

        # -- profiled iterations --
        label = f"TRITON_SUM_N{N // 1024 // 1024}M_bs{BLOCK_SIZE}_iters{PROFILE_ITERS}"
        nvtx.range_push(label)
        for _ in range(PROFILE_ITERS):
            c = triton_add(a, b)
        torch.cuda.synchronize()
        nvtx.range_pop()

        grids = triton.cdiv(N, BLOCK_SIZE)
        print(f"  [triton] N={N:>10,}  grid=({grids},)  sum[0]={c[0].item():.4f}")

    cuda_profiler.stop()    # nsys capture window ends here

    print()
    print("Done — Triton optimized sum complete.")


if __name__ == "__main__":
    main()
