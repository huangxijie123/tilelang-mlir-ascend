import tilelang
import tilelang.language as T
import torch
import os
import torch.nn.functional as F


def _batch_norm_kernel(C, L, eps, dtype):

    @tilelang.jit(out_idx=[3], target="npuir")
    def _func(block_c):

        @T.prim_func
        def main(
            x: T.Tensor[(C, L), dtype],
            weight: T.Tensor[(C), dtype],
            bias: T.Tensor[(C), dtype],
            y: T.Tensor[(C, L), dtype],
        ):
            with T.Kernel(T.ceildiv(C, block_c), is_npu=True) as (pid_c, _):
                shared_buf = T.alloc_shared((block_c, L), dtype)
                x_local = T.alloc_fragment((block_c, L), dtype)
                x_f32 = T.alloc_fragment((block_c, L), "float32")
                acc = T.alloc_fragment((block_c, 1), "float32")
                mean = T.alloc_fragment((block_c, 1), "float32")
                rstd = T.alloc_fragment((block_c, 1), "float32")

                T.copy(x[pid_c * block_c, 0], shared_buf)
                T.copy(shared_buf, x_local)

                for i, j in T.Parallel(block_c, L):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_c):
                    mean[i, 0] = acc[i, 0] / float(L)

                for i, j in T.Parallel(block_c, L):
                    x_f32[i, j] = (x_f32[i, j] - mean[i, 0]) * (
                        x_f32[i, j] - mean[i, 0]
                    )
                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_c):
                    rstd[i, 0] = acc[i, 0] / float(L) + eps
                T.vrsqrt(rstd, rstd)

                for i, j in T.Parallel(block_c, L):
                    x_local[i, j] = (
                        T.cast(x_local[i, j], "float32") - mean[i, 0]
                    ) * rstd[i, 0] * T.cast(
                        weight[pid_c * block_c + i], "float32"
                    ) + T.cast(bias[pid_c * block_c + i], "float32")

                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_c * block_c, 0])

        return main

    return _func


def _batch_norm_kernel_high_perf(C, L, eps, dtype):

    @tilelang.jit(out_idx=[3], target="npuir")
    def _func_high_perf(block_c, block_l):

        @T.prim_func
        def high_perf(
            x: T.Tensor[(C, L), dtype],
            weight: T.Tensor[(C,), dtype],
            bias: T.Tensor[(C,), dtype],
            y: T.Tensor[(C, L), dtype],
        ):
            with T.Kernel(T.ceildiv(C, block_c)) as pid_c:
                x_tile = T.alloc_shared((block_c, block_l), "float32")
                w_tile = T.alloc_shared((block_c,), "float32")
                b_tile = T.alloc_shared((block_c,), "float32")
                y_tile = T.alloc_shared((block_c, block_l), "float32")

                acc = T.alloc_fragment((block_c, 1), "float32")
                mean_val = T.alloc_fragment((block_c, 1), "float32")
                var_val = T.alloc_fragment((block_c, 1), "float32")
                rstd = T.alloc_fragment((block_c, 1), "float32")

                T.clear(acc)
                for no in T.serial(T.ceildiv(L, block_l)):
                    d_start = no * block_l
                    T.copy(x[pid_c * block_c, d_start], x_tile)
                    T.reduce_sum(x_tile, var_val, dim=1)
                    for i in T.Parallel(block_c):
                        acc[i, 0] += var_val[i, 0]
                for i in T.Parallel(block_c):
                    mean_val[i, 0] = acc[i, 0] / float(L)

                T.clear(acc)
                for no in T.serial(T.ceildiv(L, block_l)):
                    d_start = no * block_l
                    T.copy(x[pid_c * block_c, d_start], x_tile)
                    for i, j in T.Parallel(block_c, block_l):
                        y_tile[i, j] = (x_tile[i, j] - mean_val[i, 0]) * (
                            x_tile[i, j] - mean_val[i, 0]
                        )
                    T.reduce_sum(y_tile, var_val, dim=1)
                    for i in T.Parallel(block_c):
                        acc[i, 0] += var_val[i, 0]
                for i in T.Parallel(block_c):
                    var_val[i, 0] = acc[i, 0] / float(L) + eps
                T.vrsqrt(var_val, rstd)

                T.copy(weight[pid_c * block_c], w_tile)
                T.copy(bias[pid_c * block_c], b_tile)

                for no in T.serial(T.ceildiv(L, block_l)):
                    d_start = no * block_l
                    T.copy(x[pid_c * block_c, d_start], x_tile)

                    for i, j in T.Parallel(block_c, block_l):
                        y_tile[i, j] = (x_tile[i, j] - mean_val[i, 0]) * rstd[
                            i, 0
                        ] * w_tile[i] + b_tile[i]
                    T.copy(y_tile, y[pid_c * block_c, d_start])

        return high_perf

    return _func_high_perf


def batch_norm_ref(x, rm, rv, weight, bias, eps):
    return F.batch_norm(
        x.float().T,
        rm,
        rv,
        weight=weight.float(),
        bias=bias.float(),
        training=True,
        eps=eps,
    ).T.to(x.dtype)


def run_test(
    C=2048,
    L=4096,
    block_c=64,
    block_l=64,
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

    x = torch.randn((C, L), dtype=torch_dtype, device=device)
    rm = torch.zeros((C), dtype=torch_dtype, device=device)
    rv = torch.zeros((C), dtype=torch_dtype, device=device)
    weight = torch.randn((C), dtype=torch_dtype, device=device)
    bias = torch.randn((C), dtype=torch_dtype, device=device)

    y_ref = batch_norm_ref(x, rm, rv, weight, bias, eps)
    program = _batch_norm_kernel_high_perf(C, L, eps, dtype)
    y = program(block_c, block_l)(x, weight, bias)

    torch.testing.assert_close(y.float(), y_ref.float(), atol=atol, rtol=rtol)
    print("\033[32;1mPass!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()
