# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp
import shutil

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

# Cube fractal-friendly tile.
M = 16
N = 16
K = 16


def nd2nz_nz2nd(M, N, K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                A_L1 = T.alloc_L1((M, K), dtype)
                B_L1 = T.alloc_L1((K, N), dtype)
                C_L0C = T.alloc_L0C((M, N), accum_dtype)

                # GM -> L1 (NZ)
                T.npuir_load_nd2nz(A, A_L1)
                T.npuir_load_nd2nz(B, B_L1)
                # Write L0C before store (avoids read-before-write / 2d fixpipe).
                T.npuir_dot(A_L1, B_L1, C_L0C, initC=True)
                # A5: npuir_store_nz2nd -> fixpipe {dma_mode=nz2nd} (L0C -> GM)
                T.npuir_store_nz2nd(C_L0C, C)

    return main


def test_nd2nz_nz2nd():
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"
    func = nd2nz_nz2nd(M, N, K)
    kernel = tilelang.engine.lower(func, target="npuir")

    curr_name = os.path.splitext(os.path.basename(__file__))[0][5:] + ".mlir"
    output_file = "./output/" + curr_name
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(kernel)

    ref_file = "./mlir_files/" + curr_name
    if os.environ.get("UPDATE_REF") == "1":
        shutil.copyfile(output_file, ref_file)
        return

    are_identical = filecmp.cmp(output_file, ref_file, shallow=False)
    assert are_identical, f"'{output_file}' and '{ref_file}' are not identical"


if __name__ == "__main__":
    test_nd2nz_nz2nd()
