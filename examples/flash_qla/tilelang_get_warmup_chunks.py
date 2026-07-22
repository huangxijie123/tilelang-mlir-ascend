import os

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


@tilelang.jit(target="npuir")
def tilelang_get_warmup_chunks(
    num_heads,
    chunk_size,
    threshold,
    accum_dtype,
    g_dtype,
    mask_dtype,
    seqlen_dtype,
    reverse,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_cu_seqlens = T.dynamic("num_cu_seqlens")

    @T.prim_func
    def tilelang_get_warmup_chunks_kernel(
        g: T.Tensor([1, num_tokens, num_heads], dtype=g_dtype),
        ht_mask: T.Tensor([batch_size], dtype=mask_dtype),
        cu_seqlens: T.Tensor([num_cu_seqlens], dtype=seqlen_dtype),
        num_warmup_chunks: T.Tensor([batch_size, num_heads], dtype=seqlen_dtype),
        fallback_mask: T.Tensor([batch_size, num_heads], dtype=mask_dtype),
    ):
        with T.Kernel(batch_size, is_npu=True) as (bb, _):
            ht_mask_shared = T.alloc_shared((1,), mask_dtype)
            T.copy(ht_mask[bb], ht_mask_shared)
            n_fragment = T.alloc_fragment((num_heads,), seqlen_dtype)
            f_fragment = T.alloc_fragment((num_heads,), "float16")

            if ht_mask_shared[0] == 0:
                g_shared = T.alloc_shared((num_heads,), g_dtype)
                g_cumsum = T.alloc_fragment((num_heads,), accum_dtype)

                sentinel = T.alloc_fragment((num_heads,), seqlen_dtype)
                cmp_lt = T.alloc_fragment((num_heads,), "bool")
                cmp_eq = T.alloc_fragment((num_heads,), "bool")
                cond = T.alloc_fragment((num_heads,), "bool")
                tmp1 = T.alloc_fragment((num_heads,), seqlen_dtype)

                f_cmp = T.alloc_fragment((num_heads,), "bool")
                zero_full = T.alloc_fragment((num_heads,), "float16")
                one_full = T.alloc_fragment((num_heads,), "float16")

                sentinel_f32 = T.alloc_fragment((num_heads,), "float32")
                n_fragment_f32 = T.alloc_fragment((num_heads,), "float32")

                T.clear(g_cumsum)

                num_iters = (cu_seqlens[bb + 1] - cu_seqlens[bb]) // chunk_size
                sentinel_val = num_iters + 1
                T.vbrc(sentinel_val, sentinel)
                T.vbrc(sentinel_val, n_fragment)
                T.vbrc(T.float32(sentinel_val), sentinel_f32)
                T.vbrc(T.int32(0), tmp1)

                start_idx = cu_seqlens[bb] + chunk_size - 1
                end_idx = cu_seqlens[bb + 1] - 1
                for i_s in T.serial(num_iters):
                    if reverse:
                        row_idx = start_idx + i_s * chunk_size
                        T.copy(g[0, row_idx, 0:num_heads], g_shared)
                    else:
                        row_idx = end_idx - i_s * chunk_size
                        T.copy(g[0, row_idx, 0:num_heads], g_shared)

                    if g_dtype != accum_dtype:
                        g_shared_cast = T.alloc_shared((num_heads,), accum_dtype)
                        T.vcast(g_shared, g_shared_cast)
                        T.vadd(g_cumsum, g_shared_cast, g_cumsum)
                    else:
                        T.vadd(g_cumsum, g_shared, g_cumsum)

                    T.vcmp(g_cumsum, T.float32(threshold), cmp_lt, "lt")
                    T.vcmp(n_fragment, sentinel, cmp_eq, "eq")
                    T.vand(cmp_lt, cmp_eq, cond)
                    T.vadd(tmp1, 1, tmp1)
                    T.vselect(cond, tmp1, n_fragment, n_fragment)

                T.vbrc(T.float16(0), zero_full)
                T.vbrc(T.float16(1), one_full)
                T.vcast(n_fragment, n_fragment_f32)
                T.vcmp(n_fragment_f32, sentinel_f32, f_cmp, "lt")
                T.vselect(f_cmp, zero_full, one_full, f_fragment)

                T.vbrc(num_iters, tmp1)
                T.vselect(f_cmp, n_fragment, tmp1, n_fragment)

                T.copy(n_fragment, num_warmup_chunks[bb, 0])
                T.copy(f_fragment, fallback_mask[bb, 0])
            else:
                T.vbrc(T.int32(0), n_fragment)
                T.copy(n_fragment, num_warmup_chunks[bb, 0])

    return tilelang_get_warmup_chunks_kernel


def get_warmup_chunks(
    g: torch.Tensor,  # [1, num_total_tokens, num_v_heads]
    cu_seqlens: torch.Tensor,  # [cp_real_batch_size + 1]
    ht_mask: torch.Tensor,  # [cp_real_batch_size]
    chunk_size: int = 64,
    threshold: float = -10.0,
    reverse: bool = False,
):
    batch_size, num_tokens, num_heads = g.shape
    real_batch_size = ht_mask.shape[0]
    assert cu_seqlens.shape[0] == real_batch_size + 1
    assert batch_size == 1
    assert chunk_size == 64

    tilelang_get_warmup_chunks_kernel = tilelang_get_warmup_chunks(
        num_heads=num_heads,
        chunk_size=chunk_size,
        threshold=threshold,
        accum_dtype="float32",
        g_dtype=DTYPE_TO_STR[g.dtype],
        mask_dtype=DTYPE_TO_STR[ht_mask.dtype],
        seqlen_dtype=DTYPE_TO_STR[cu_seqlens.dtype],
        reverse=reverse,
    )
    num_warmup_chunks = torch.empty(
        [real_batch_size, num_heads], dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    fallback_mask = torch.empty(
        [real_batch_size, num_heads], dtype=ht_mask.dtype, device=cu_seqlens.device
    )
    tilelang_get_warmup_chunks_kernel(
        g, ht_mask, cu_seqlens, num_warmup_chunks, fallback_mask
    )

    return num_warmup_chunks, fallback_mask


def get_warmup_chunks_torch_ref(
    g, ht_mask, cu_seqlens, num_heads, chunk_size, threshold, reverse
):
    batch_size = ht_mask.shape[0]
    num_warmup_chunks = torch.zeros(
        (batch_size, num_heads), dtype=cu_seqlens.dtype, device="cpu"
    )
    fallback_mask = torch.zeros(
        (batch_size, num_heads), dtype=ht_mask.dtype, device="cpu"
    )

    g_cpu = g.cpu()
    ht_mask_cpu = ht_mask.cpu()
    cu_seqlens_cpu = cu_seqlens.cpu()

    n_iters_list = [0] * batch_size
    for b in range(batch_size):
        if ht_mask_cpu[b] != 0:
            num_warmup_chunks[b, :] = 0
            fallback_mask[b, :] = 0
        else:
            seq_start = cu_seqlens_cpu[b].item()
            seq_end = cu_seqlens_cpu[b + 1].item()
            n_iters = (seq_end - seq_start) // chunk_size
            n_iters_list[b] = n_iters

            g_cumsum = torch.zeros(num_heads, dtype=torch.float32)
            n_frag = torch.full((num_heads,), n_iters, dtype=cu_seqlens.dtype)
            f_frag = torch.ones(num_heads, dtype=ht_mask.dtype)

            for s in range(n_iters):
                idx = seq_end - s * chunk_size - 1
                if reverse:
                    idx = seq_start + (s + 1) * chunk_size - 1

                g_val = g_cpu[0, idx, :]
                g_cumsum += g_val
                for h in range(num_heads):
                    if g_cumsum[h] < threshold and n_frag[h] == n_iters:
                        n_frag[h] = s + 1
                        f_frag[h] = 0

            num_warmup_chunks[b, :] = n_frag
            fallback_mask[b, :] = f_frag
    print(n_iters_list)
    return num_warmup_chunks, fallback_mask


def test_get_warmup_chunks():
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    tilelang.cache.clear_cache()

    batch_size = 48
    num_tokens = 1024 * 128
    num_heads = 256
    chunk_size = 64
    threshold = -10.0

    torch.manual_seed(42)

    g = (torch.randn((1, num_tokens, num_heads), dtype=torch.float16) * 0.8).npu()
    ht_mask = torch.randint(0, 2, (batch_size,), dtype=torch.int8).npu()

    num_chunks = num_tokens // chunk_size
    rand_boundaries = torch.sort(torch.randint(1, num_chunks, (batch_size - 1,)))[0]
    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            (rand_boundaries * chunk_size).to(torch.int32),
            torch.tensor([num_tokens], dtype=torch.int32),
        ]
    ).npu()

    reverse = True
    ref_num_warmup, ref_fallback = get_warmup_chunks_torch_ref(
        g, ht_mask, cu_seqlens, num_heads, chunk_size, threshold, reverse
    )

    num_warmup_chunks_out, fallback_mask_out = get_warmup_chunks(
        g, cu_seqlens, ht_mask, chunk_size, threshold, reverse
    )

    torch.testing.assert_close(
        num_warmup_chunks_out.cpu(), ref_num_warmup, rtol=0, atol=0
    )
    print("num_warmup_chunks check passed!")

    torch.testing.assert_close(
        fallback_mask_out.cpu().to(ref_fallback.dtype), ref_fallback, rtol=0, atol=0
    )
    print("fallback_mask check passed!")

    for _ in range(10):
        num_warmup_chunks_out, fallback_mask_out = get_warmup_chunks(
            g, cu_seqlens, ht_mask, chunk_size, threshold
        )

    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    test_get_warmup_chunks()
