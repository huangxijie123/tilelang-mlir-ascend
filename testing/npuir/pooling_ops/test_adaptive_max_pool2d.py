# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""AdaptiveMaxPool2d operator test (Developer mode)."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("adaptive_max_pool2d"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]


def adaptive_max_pool2d_kernel(H_in, W_in, H_out, W_out, dtype="float16"):
    """Adaptive max pool 2d kernel (uniform-split strategy).

    Preconditions: H_in % H_out == 0 and W_in % W_out == 0.
    Under these conditions every output element covers an identical
    stride_h × stride_w input window, and the kernel degenerates to a
    standard max pool with stride = kernel = H_in // H_out.
    """
    assert H_in % H_out == 0, (
        f"H_in ({H_in}) must be divisible by H_out ({H_out}) for uniform-split strategy"
    )
    assert W_in % W_out == 0, (
        f"W_in ({W_in}) must be divisible by W_out ({W_out}) for uniform-split strategy"
    )
    stride_h = H_in // H_out
    stride_w = W_in // W_out

    @T.prim_func
    def kernel_fn(
        In: T.Tensor((H_in, W_in), dtype),
        Out: T.Tensor((H_out, W_out), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            # --- shared-memory buffers ---
            in_sh = T.alloc_shared((H_in, W_in), dtype)
            out_sh = T.alloc_shared((H_out, W_out), dtype)

            # 1. Load entire input
            T.copy(In, in_sh)

            # 2. Window max (no if inside T.serial)
            for oi, oj in T.Parallel(H_out, W_out):
                hi = oi * stride_h
                wi = oj * stride_w

                # ---- initialise with window element (0, 0) ----
                out_sh[oi, oj] = in_sh[hi, wi]

                # ---- remaining window elements ----
                for kj in T.serial(stride_w - 1):
                    out_sh[oi, oj] = T.max(out_sh[oi, oj], in_sh[hi, wi + kj + 1])
                for ki in T.serial(stride_h - 1):
                    for kj in T.serial(stride_w):
                        out_sh[oi, oj] = T.max(
                            out_sh[oi, oj], in_sh[hi + ki + 1, wi + kj]
                        )

            # 3. Store
            T.copy(out_sh, Out)

    return kernel_fn


def _ref_adaptive_max_pool2d(x: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
    """Reference via ``torch.nn.functional.adaptive_max_pool2d``.

    F.adaptive_max_pool2d expects (N, C, H, W) input; we lift the 2-d
    input accordingly and squeeze the result back.  The reference is
    computed in float32 and cast back to avoid internal fp16 rounding
    differences in torch.
    """
    x_4d = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    ref_4d = F.adaptive_max_pool2d(x_4d.float(), (H_out, W_out))
    return ref_4d[0, 0].to(x.dtype)


def _compile_and_run(H_in, W_in, H_out, W_out, dtype, rtol=1e-2, atol=1e-2):
    dtype_t = getattr(torch, dtype)
    x = gen_tensor((H_in, W_in), dtype, kind="randn")
    out = torch.zeros((H_out, W_out), dtype=dtype_t, device="npu")
    ref = _ref_adaptive_max_pool2d(x.cpu(), H_out, W_out)

    func = adaptive_max_pool2d_kernel(
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        dtype=dtype,
    )
    compiled = tilelang.compile(func, target="npuir")
    compiled(x, out)

    assert_close(out.cpu(), ref, dtype=dtype, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_max_pool2d_4x4(dtype):
    """16×16 → 4×4  (4×4 window)."""
    _compile_and_run(H_in=16, W_in=16, H_out=4, W_out=4, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_max_pool2d_8x8(dtype):
    """32×32 → 8×8  (4×4 window)."""
    _compile_and_run(H_in=32, W_in=32, H_out=8, W_out=8, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_max_pool2d_6x6(dtype):
    """24×24 → 6×6  (4×4 window)."""
    _compile_and_run(H_in=24, W_in=24, H_out=6, W_out=6, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_max_pool2d_3x6(dtype):
    """12×24 → 3×6  (rectangular input: 4×4 window)."""
    _compile_and_run(H_in=12, W_in=24, H_out=3, W_out=6, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_adaptive_max_pool2d_2x2(dtype):
    """16×16 → 2×2  (8×8 window, larger kernel)."""
    _compile_and_run(H_in=16, W_in=16, H_out=2, W_out=2, dtype=dtype)
