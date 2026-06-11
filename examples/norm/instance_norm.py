import torch
import os

from group_norm import group_norm_ref as instance_norm_ref
from group_norm import _group_norm_kernel_high_perf as _instance_norm_kernel_high_perf


def run_test(
    N=2048,
    C=1,
    block_m=64,
    block_n=1,
    eps=1e-5,
    dtype="float16",
    device="npu",
    atol=1e-2,
    rtol=1e-2,
    g=64,
):

    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]

    x = torch.zeros((N, C), dtype=torch_dtype, device=device)
    weight = torch.randn((C,), dtype=torch_dtype, device=device)
    bias = torch.randn((C,), dtype=torch_dtype, device=device)

    y_ref = instance_norm_ref(x, weight, bias, g, eps)
    program = _instance_norm_kernel_high_perf(N, C, eps, dtype)
    y = program(block_m, block_n)(x, weight, bias)

    torch.testing.assert_close(y.float(), y_ref.float(), atol=atol, rtol=rtol)
    print("\033[32;1mPass!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()
