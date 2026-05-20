import tilelang
import tilelang.language as T
import torch
import os
import torch.nn.functional as F

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _rms_norm_kernel(M, N, eps, dtype):
    N_padded = _align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[2], target="npuir")
    def _func(block_m):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                xsq_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                sumsq = T.alloc_fragment((block_m, 1), "float32")
                rrms = T.alloc_fragment((block_m, 1), "float32")

                # Load input row block
                T.copy(x[pid_m * block_m, 0], shared_buf)
                T.copy(shared_buf, x_local)

                # Compute x^2 in fp32
                for i, j in T.Parallel(block_m, N_padded):
                    xsq_f32[i, j] = (
                        T.cast(x_local[i, j], "float32") * T.cast(x_local[i, j], "float32")
                    )

                # Sum of squares along hidden dim
                T.reduce_sum(xsq_f32, sumsq, dim=1)

                # rrms = rsqrt(mean(x^2) + eps), using original N (not padded)
                for i in T.Parallel(block_m):
                    rrms[i, 0] = sumsq[i, 0] / float(N) + eps
                T.vrsqrt(rrms, rrms)

                # y = x * rrms * weight, result stored back in x_local
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = T.cast((
                        T.cast(x_local[i, j], "float32") * rrms[i, 0] * T.cast(weight[j], "float32")
                    ), "float16")

                # Write output
                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func

def rms_norm_ref(x, weight, eps):
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((x_f32 / rms) * weight.float()).to(x.dtype)


def run_test(
    M=4096,
    N=4096,
    block_m=4,
    eps=1e-5,
    dtype="float16",
    device="npu",
    atol=1e-2,
    rtol=1e-2,
):
    n_padded = _align_up(N, ALIGNMENT)

    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]

    x = torch.zeros((M, n_padded), dtype=torch_dtype, device=device)
    x[:, :N] = torch.randn((M, N), dtype=torch_dtype, device=device)
    weight = torch.randn((n_padded,), dtype=torch_dtype, device=device)
   
    y_ref = rms_norm_ref(x, weight, eps)
    program = _rms_norm_kernel(M, N, eps, dtype)
    y = program(block_m)(x, weight)

    torch.testing.assert_close(y.float(), y_ref.float(), atol=atol, rtol=rtol)
    print("\033[32;1mPass!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()