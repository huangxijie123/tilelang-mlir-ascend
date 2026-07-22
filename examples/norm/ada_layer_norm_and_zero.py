"""Adaptive LayerNorm (AdaLN / AdaLN-Zero) kernel using TileLang.

AdaLN:      y = scale * LayerNorm(x) + shift
AdaLN-Zero: y = gate * (scale * LayerNorm(x) + shift)

The `has_gate` parameter controls the variant:
- has_gate=False → AdaLN
- has_gate=True  → AdaLN-Zero

"""

import tilelang
import tilelang.language as T
import torch

import os
import torch.nn.functional as F


def _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False):
    N_inv = 1.0 / float(N)

    @tilelang.jit(out_idx=[3] if not has_gate else [4], target="npuir")
    def _func(block_m, block_n):
        if not has_gate:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, N), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                    x_tile = T.alloc_ub((block_m, block_n), "float32")
                    scale_tile = T.alloc_ub((block_m, block_n), "float32")

                    mean_val = T.alloc_ub((block_m, 1), "float32")
                    var_val = T.alloc_ub((block_m, 1), "float32")
                    red_tmp = T.alloc_ub((block_m, 1), "float32")
                    rstd = T.alloc_ub((block_m, 1), "float32")

                    T.clear(mean_val)
                    T.clear(var_val)

                    for no in T.serial(T.ceildiv(N, block_n)):
                        offset_m = pid_m * block_m
                        tile_size_m = T.min(block_m, M - pid_m * block_m)
                        offset_n = no * block_n
                        remain_n = T.min(block_n, N - offset_n)

                        T.copy(
                            x[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            x_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.npuir_reduce(
                            x_tile,
                            mean_val,
                            dims=1,
                            reduce_mode="sum",
                            size=[tile_size_m, remain_n],
                            clear=False,
                        )

                        T.vmul(x_tile, x_tile, x_tile)
                        T.npuir_reduce(
                            x_tile,
                            var_val,
                            dims=1,
                            reduce_mode="sum",
                            size=[tile_size_m, remain_n],
                            clear=False,
                        )

                    T.vmul(mean_val, N_inv, mean_val)
                    T.vmul(var_val, N_inv, var_val)

                    T.vmul(mean_val, mean_val, red_tmp)
                    T.vsub(var_val, red_tmp, var_val)
                    T.vadd(var_val, eps, var_val)
                    T.vrsqrt(var_val, rstd)

                    for no in T.serial(T.ceildiv(N, block_n)):
                        offset_m = pid_m * block_m
                        tile_size_m = T.min(block_m, M - pid_m * block_m)
                        offset_n = no * block_n
                        remain_n = T.min(block_n, N - offset_n)

                        T.copy(
                            x[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            x_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.vsub(x_tile, mean_val, x_tile)
                        T.vmul(x_tile, rstd, x_tile)

                        T.copy(
                            scale[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            scale_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.vmul(x_tile, scale_tile, x_tile)

                        T.copy(
                            shift[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            scale_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.vadd(x_tile, scale_tile, x_tile)
                        T.copy(
                            x_tile[0:tile_size_m, 0:remain_n],
                            y[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                        )

            return main
        else:

            @T.prim_func
            def main_gated(
                x: T.Tensor[(M, N), dtype],
                scale: T.Tensor[(M, N), dtype],
                shift: T.Tensor[(M, N), dtype],
                gate: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, N), dtype],
            ):

                with T.Kernel(
                    T.ceildiv(M, block_m),
                    is_npu=True,
                ) as (pid_m, _):
                    x_tile = T.alloc_ub((block_m, block_n), "float32")
                    scale_tile = T.alloc_ub((block_m, block_n), "float32")

                    mean_val = T.alloc_ub((block_m, 1), "float32")
                    var_val = T.alloc_ub((block_m, 1), "float32")
                    red_tmp = T.alloc_ub((block_m, 1), "float32")
                    rstd = T.alloc_ub((block_m, 1), "float32")

                    T.clear(mean_val)
                    T.clear(var_val)

                    for no in T.serial(T.ceildiv(N, block_n)):
                        offset_m = pid_m * block_m
                        tile_size_m = T.min(block_m, M - pid_m * block_m)
                        offset_n = no * block_n
                        remain_n = T.min(block_n, N - offset_n)

                        T.copy(
                            x[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            x_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.npuir_reduce(
                            x_tile,
                            mean_val,
                            dims=1,
                            reduce_mode="sum",
                            size=[tile_size_m, remain_n],
                            clear=False,
                        )

                        T.vmul(x_tile, x_tile, x_tile)
                        T.npuir_reduce(
                            x_tile,
                            var_val,
                            dims=1,
                            reduce_mode="sum",
                            size=[tile_size_m, remain_n],
                            clear=False,
                        )

                    T.vmul(mean_val, N_inv, mean_val)
                    T.vmul(var_val, N_inv, var_val)

                    T.vmul(mean_val, mean_val, red_tmp)
                    T.vsub(var_val, red_tmp, var_val)
                    T.vadd(var_val, eps, var_val)
                    T.vrsqrt(var_val, rstd)

                    for no in T.serial(T.ceildiv(N, block_n)):
                        offset_m = pid_m * block_m
                        tile_size_m = T.min(block_m, M - pid_m * block_m)
                        offset_n = no * block_n
                        remain_n = T.min(block_n, N - offset_n)

                        T.copy(
                            x[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            x_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.vsub(x_tile, mean_val, x_tile)
                        T.vmul(x_tile, rstd, x_tile)

                        T.copy(
                            scale[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            scale_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.vmul(x_tile, scale_tile, x_tile)

                        T.copy(
                            shift[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            scale_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.vadd(x_tile, scale_tile, x_tile)

                        T.copy(
                            gate[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                            scale_tile[0:tile_size_m, 0:remain_n],
                        )
                        T.vmul(x_tile, scale_tile, x_tile)

                        T.copy(
                            x_tile[0:tile_size_m, 0:remain_n],
                            y[
                                offset_m : offset_m + tile_size_m,
                                offset_n : offset_n + remain_n,
                            ],
                        )

            return main_gated

    return _func


def ada_layer_norm_ref(x, scale, shift, eps):
    N = x.shape[-1]

    y = F.layer_norm(
        x.float(),
        # x,
        (N,),
        weight=None,
        bias=None,
        eps=eps,
    )

    y = y * scale.float() + shift.float()
    return y.to(x.dtype)


def ada_layer_norm_zero_ref(x, scale, shift, gate, eps):
    N = x.shape[-1]

    y = F.layer_norm(
        x.float(),
        (N,),
        weight=None,
        bias=None,
        eps=eps,
    )

    y = gate.float() * (y * scale.float() + shift.float())
    return y.to(x.dtype)


def run_adaln_test(
    M=1024,
    N=16384,
    block_m=2,
    block_n=2048,
    eps=1e-5,
    dtype="float16",
    device="npu",
    atol=1e-2,
    rtol=1e-2,
):
    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]

    # Inputs
    x = torch.zeros(
        (M, N),
        dtype=torch_dtype,
        device=device,
    )
    x[:, :N] = torch.randn((M, N), dtype=torch_dtype, device=device)
    scale = torch.randn((M, N), dtype=torch_dtype, device=device)
    shift = torch.randn(
        (M, N),
        dtype=torch_dtype,
        device=device,
    )

    # Reference
    y_ref = ada_layer_norm_ref(x, scale, shift, eps)

    program = _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=False)
    y = program(block_m, block_n)(x, scale, shift)

    torch.testing.assert_close(
        y[:M, :N].float(),
        y_ref[:M, :N].float(),
        atol=atol,
        rtol=rtol,
    )
    print("\033[32;1mAdaLN Pass!\033[0m")


def run_adaln_zero_test(
    M=4096,
    N=4096,
    block_m=4,
    block_n=4096,
    eps=1e-5,
    dtype="float16",
    device="npu",
    atol=1e-2,
    rtol=1e-2,
):
    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]

    # Inputs
    x = torch.zeros(
        (M, N),
        dtype=torch_dtype,
        device=device,
    )
    x[:, :N] = torch.randn(
        (M, N),
        dtype=torch_dtype,
        device=device,
    )
    scale = torch.randn(
        (M, N),
        dtype=torch_dtype,
        device=device,
    )
    shift = torch.randn(
        (M, N),
        dtype=torch_dtype,
        device=device,
    )
    gate = torch.randn(
        (M, N),
        dtype=torch_dtype,
        device=device,
    )
    gate = torch.clamp(gate, min=-1.0, max=1.0)

    # Reference
    y_ref = ada_layer_norm_zero_ref(x, scale, shift, gate, eps)

    # TileLang kernel
    program = _ada_layer_norm_kernel(M, N, eps, dtype, has_gate=True)
    y = program(block_m, block_n)(x, scale, shift, gate)

    # Compare
    torch.testing.assert_close(
        y[:M, :N].float(),
        y_ref[:M, :N].float(),
        atol=atol,
        rtol=rtol,
    )
    print("\033[32;1mAdaLN_Zero Pass!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    tilelang.cache.clear_cache()
    run_adaln_test()
    run_adaln_zero_test()
