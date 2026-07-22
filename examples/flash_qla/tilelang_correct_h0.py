import os

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

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
def tilelang_correct_h0(
    H,
    DK,
    DV,
    res_dtype,
    accum_dtype,
    buffer_dtype,
    seqlen_dtype,
    mask_dtype,
    use_raw_h0,
    state_v_first,
    reverse: bool = False,
    transpose_m: bool = False,
    block_DV: int = 64,
):
    zero_init = not use_raw_h0
    cp_batch_size = T.dynamic("cp_batch_size")
    num_seq_map = T.dynamic("num_seq_map")
    raw_batch_size = T.dynamic("raw_batch_size")

    state_shape = (
        (cp_batch_size, H, DV, DK) if state_v_first else (cp_batch_size, H, DK, DV)
    )
    raw_state_shape = (
        (raw_batch_size, H, DV, DK) if state_v_first else (raw_batch_size, H, DK, DV)
    )

    assert DV % block_DV == 0, (
        f"block_DV={block_DV} must divide DV={DV} exactly, "
        f"got remainder {DV % block_DV}"
    )
    num_dv_blocks = DV // block_DV
    h_frag_shape = (block_DV, DK) if state_v_first else (DK, block_DV)
    need_init_vcast = res_dtype != accum_dtype
    need_loop_vcast = buffer_dtype != accum_dtype

    @T.macro
    def kernel_body(
        bh,
        bv,
        seq_start_idx,
        seq_end_idx,
        num_iters,
        ht_buffer,
        mt_buffer,
        fallback_mask,
        cp_h0,
        h_fragment,
    ):
        h_shared = T.alloc_shared(h_frag_shape, dtype=buffer_dtype)
        hd_shared = T.alloc_shared(h_frag_shape, dtype=buffer_dtype)
        m_shared = T.alloc_shared((DK, DK), dtype=buffer_dtype)
        acc = T.alloc_fragment(h_frag_shape, dtype=accum_dtype)

        DV_start = bv * block_DV
        DV_end = (bv + 1) * block_DV

        for i_s in T.serial(num_iters - 1):
            idx = (
                seq_start_idx + num_iters - 1 - i_s if reverse else seq_start_idx + i_s
            )

            if state_v_first:
                T.copy(h_fragment, cp_h0[idx, bh, DV_start:DV_end, 0:DK])
            else:
                T.copy(h_fragment, cp_h0[idx, bh, 0:DK, DV_start:DV_end])

            if need_loop_vcast:
                T.vcast(h_fragment, hd_shared, round_mode="rint")
            else:
                T.copy(h_fragment, hd_shared)

            if state_v_first:
                T.copy(ht_buffer[idx, bh, DV_start:DV_end, 0:DK], h_shared)
            else:
                T.copy(ht_buffer[idx, bh, 0:DK, DV_start:DV_end], h_shared)

            if need_loop_vcast:
                T.vcast(h_shared, h_fragment, round_mode="rint")
            else:
                T.copy(h_shared, h_fragment)

            if fallback_mask[idx, bh] != 0:
                T.copy(mt_buffer[idx, bh, 0:DK, 0:DK], m_shared)

                if state_v_first:
                    if transpose_m:
                        T.gemm(
                            hd_shared,
                            m_shared,
                            acc,
                            size=[block_DV, DK, DK],
                            initC=True,
                        )
                    else:
                        T.gemm(
                            hd_shared,
                            m_shared,
                            acc,
                            size=[block_DV, DK, DK],
                            initC=True,
                            b_transpose=True,
                        )
                else:
                    if transpose_m:
                        T.gemm(
                            m_shared,
                            hd_shared,
                            acc,
                            size=[DK, DK, block_DV],
                            initC=True,
                            a_transpose=True,
                        )
                    else:
                        T.gemm(
                            m_shared,
                            hd_shared,
                            acc,
                            size=[DK, DK, block_DV],
                            initC=True,
                        )

                T.vadd(h_fragment, acc, h_fragment)

        last_idx = seq_start_idx if reverse else seq_start_idx + num_iters - 1

        if state_v_first:
            T.copy(h_fragment, cp_h0[last_idx, bh, DV_start:DV_end, 0:DK])
        else:
            T.copy(h_fragment, cp_h0[last_idx, bh, 0:DK, DV_start:DV_end])

    @T.prim_func
    def tilelang_correct_h0_kernel(
        raw_h0: T.Tensor(raw_state_shape, dtype=res_dtype),
        ht_buffer: T.Tensor(state_shape, dtype=buffer_dtype),
        mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=buffer_dtype),
        fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
        seq_map_r2c: T.Tensor([num_seq_map], dtype=seqlen_dtype),
        cp_h0: T.Tensor(state_shape, dtype=res_dtype),
    ):
        _ = T.meta_var(seqlen_dtype)
        _ = T.meta_var(mask_dtype)
        with T.Kernel(
            num_dv_blocks * H * raw_batch_size,
            is_npu=True,
        ) as (bbhv, _):
            bbh = bbhv // num_dv_blocks
            bv = bbhv % num_dv_blocks
            bb = bbh // H
            bh = bbh % H

            seq_start_idx = seq_map_r2c[bb]
            seq_end_idx = seq_map_r2c[bb + 1]
            num_iters = seq_end_idx - seq_start_idx

            h_fragment = T.alloc_fragment(h_frag_shape, dtype=accum_dtype)

            if zero_init:
                T.clear(h_fragment)
            else:
                h_init = T.alloc_shared(h_frag_shape, dtype=res_dtype)

                if state_v_first:
                    T.copy(
                        raw_h0[
                            bb,
                            bh,
                            bv * block_DV : (bv + 1) * block_DV,
                            0:DK,
                        ],
                        h_init,
                    )
                else:
                    T.copy(
                        raw_h0[
                            bb,
                            bh,
                            0:DK,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                        h_init,
                    )

                if need_init_vcast:
                    T.vcast(h_init, h_fragment, round_mode="rint")
                else:
                    T.copy(h_init, h_fragment)

            kernel_body(
                bh,
                bv,
                seq_start_idx,
                seq_end_idx,
                num_iters,
                ht_buffer,
                mt_buffer,
                fallback_mask,
                cp_h0,
                h_fragment,
            )

    return tilelang_correct_h0_kernel


def correct_initial_states(
    raw_h0,
    ht_buffer,
    mt_buffer,
    fallback_mask,
    seq_map_r2c,
    state_v_first: bool = False,
    reverse: bool = False,
    transpose_m: bool = False,
):
    cp_batch_size = fallback_mask.shape[0]
    _, num_heads, dim_2, dim_3 = ht_buffer.shape
    if state_v_first:
        v_head_dim, k_head_dim = dim_2, dim_3
    else:
        k_head_dim, v_head_dim = dim_2, dim_3

    assert k_head_dim == v_head_dim == 128
    assert ht_buffer.dtype == torch.float16, (
        f"buffer_dtype must be float16 for T.gemm on npuir, got {ht_buffer.dtype}"
    )
    assert mt_buffer.dtype == ht_buffer.dtype, (
        f"mt_buffer dtype {mt_buffer.dtype} must match ht_buffer dtype {ht_buffer.dtype}"
    )

    raw_batch_size = seq_map_r2c.shape[0] - 1

    if raw_h0 is None:
        res_dtype = torch.float32
        use_raw_h0 = False
        if state_v_first:
            raw_h0 = torch.zeros(
                raw_batch_size,
                num_heads,
                v_head_dim,
                k_head_dim,
                dtype=res_dtype,
                device=ht_buffer.device,
            )
        else:
            raw_h0 = torch.zeros(
                raw_batch_size,
                num_heads,
                k_head_dim,
                v_head_dim,
                dtype=res_dtype,
                device=ht_buffer.device,
            )
    else:
        res_dtype = raw_h0.dtype
        use_raw_h0 = True

    kernel = tilelang_correct_h0(
        H=num_heads,
        DK=k_head_dim,
        DV=v_head_dim,
        res_dtype=DTYPE_TO_STR[res_dtype],
        accum_dtype="float32",
        buffer_dtype=DTYPE_TO_STR[ht_buffer.dtype],
        seqlen_dtype=DTYPE_TO_STR[seq_map_r2c.dtype],
        mask_dtype=DTYPE_TO_STR[fallback_mask.dtype],
        state_v_first=state_v_first,
        use_raw_h0=use_raw_h0,
        reverse=reverse,
        transpose_m=transpose_m,
    )

    if state_v_first:
        cp_h0 = torch.empty(
            (cp_batch_size, num_heads, v_head_dim, k_head_dim),
            dtype=res_dtype,
            device=ht_buffer.device,
        )
    else:
        cp_h0 = torch.empty(
            (cp_batch_size, num_heads, k_head_dim, v_head_dim),
            dtype=res_dtype,
            device=ht_buffer.device,
        )

    kernel(raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c, cp_h0)

    return cp_h0


def correct_terminal_states(
    raw_dht,
    dht_buffer,
    mt_buffer,
    fallback_mask,
    seq_map_r2c,
    state_v_first: bool = False,
):
    return correct_initial_states(
        raw_dht,
        dht_buffer,
        mt_buffer,
        fallback_mask,
        seq_map_r2c,
        state_v_first=state_v_first,
        reverse=True,
        transpose_m=True,
    )


def correct_h0_torch_ref(
    raw_h0,
    ht_buffer,
    mt_buffer,
    fallback_mask,
    seq_map_r2c,
    state_v_first: bool = False,
    reverse: bool = False,
    transpose_m: bool = False,
    block_DV: int = 32,
):
    cp_batch, H, dim2, dim3 = ht_buffer.shape
    if state_v_first:
        DV, DK = dim2, dim3
    else:
        DK, DV = dim2, dim3
    res_dtype = torch.float32 if raw_h0 is None else raw_h0.dtype
    raw_batch = seq_map_r2c.shape[0] - 1

    out_shape = (cp_batch, H, DV, DK) if state_v_first else (cp_batch, H, DK, DV)
    cp_h0 = torch.zeros(out_shape, dtype=res_dtype, device="cpu")
    num_iters = []
    for bb in range(raw_batch):
        s, e = int(seq_map_r2c[bb]), int(seq_map_r2c[bb + 1])
        N = e - s
        num_iters.append(N)
        for bh in range(H):
            for bv in range(DV // block_DV):
                dvs, dve = bv * block_DV, (bv + 1) * block_DV
                if raw_h0 is not None:
                    if state_v_first:
                        h = raw_h0[bb, bh, dvs:dve, 0:DK].float()
                    else:
                        h = raw_h0[bb, bh, 0:DK, dvs:dve].float()
                else:
                    sh = (block_DV, DK) if state_v_first else (DK, block_DV)
                    h = torch.zeros(sh, dtype=torch.float32)

                for i_s in range(N - 1):
                    idx = (s + N - 1 - i_s) if reverse else (s + i_s)
                    if state_v_first:
                        cp_h0[idx, bh, dvs:dve, 0:DK] = h.to(res_dtype)
                    else:
                        cp_h0[idx, bh, 0:DK, dvs:dve] = h.to(res_dtype)

                    if state_v_first:
                        h_new = ht_buffer[idx, bh, dvs:dve, 0:DK].float()
                    else:
                        h_new = ht_buffer[idx, bh, 0:DK, dvs:dve].float()
                    m = mt_buffer[idx, bh, 0:DK, 0:DK].float()

                    if fallback_mask[idx, bh].item():
                        hd = h.to(ht_buffer.dtype).float()
                        if state_v_first:
                            corr = hd @ m if transpose_m else hd @ m.t()
                        else:
                            corr = m.t() @ hd if transpose_m else m @ hd
                        h = h_new + corr.float()
                    else:
                        h = h_new

                last = s if reverse else s + N - 1
                if state_v_first:
                    cp_h0[last, bh, dvs:dve, 0:DK] = h.to(res_dtype)
                else:
                    cp_h0[last, bh, 0:DK, dvs:dve] = h.to(res_dtype)
    print(f"num_iters: {num_iters}")
    return cp_h0


def _run_case(
    use_raw_h0,
    state_v_first,
    reverse,
    transpose_m,
    fb_pattern,
    dtype,
    rtol,
    atol,
    seed=0,
    terminal=False,
):
    torch.manual_seed(seed)
    tilelang.cache.clear_cache()

    H = 4
    DK = 128
    DV = 128
    block_DV = 32
    raw_batch = 3
    num_iters_per_batch = [10, 20, 30]
    cp_batch = sum(num_iters_per_batch)

    seq_map_r2c = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(num_iters_per_batch), dim=0).tolist()),
        dtype=torch.int32,
    )
    print(f"seq_map_r2c: {seq_map_r2c}")

    if state_v_first:
        ht_buffer = torch.randn(cp_batch, H, DV, DK, dtype=dtype, device="npu")
    else:
        ht_buffer = torch.randn(cp_batch, H, DK, DV, dtype=dtype, device="npu")
    mt_scale = 0.5 / (DK**0.5)
    mt_buffer = torch.randn(cp_batch, H, DK, DK, dtype=dtype, device="npu") * mt_scale

    if fb_pattern == "all_true":
        fallback_mask = torch.ones(cp_batch, H, dtype=torch.int8, device="npu")
    elif fb_pattern == "all_false":
        fallback_mask = torch.zeros(cp_batch, H, dtype=torch.int8, device="npu")
    else:
        fallback_mask = torch.randint(
            0, 2, (cp_batch, H), dtype=torch.int8, device="npu"
        )

    if use_raw_h0:
        if state_v_first:
            raw_h0 = torch.randn(raw_batch, H, DV, DK, dtype=dtype, device="npu")
        else:
            raw_h0 = torch.randn(raw_batch, H, DK, DV, dtype=dtype, device="npu")
    else:
        raw_h0 = None

    ref = correct_h0_torch_ref(
        raw_h0.cpu() if raw_h0 is not None else None,
        ht_buffer.cpu(),
        mt_buffer.cpu(),
        fallback_mask.cpu(),
        seq_map_r2c,
        state_v_first=state_v_first,
        reverse=True if terminal else reverse,
        transpose_m=True if terminal else transpose_m,
        block_DV=block_DV,
    )

    if terminal:
        out = correct_terminal_states(
            raw_h0,
            ht_buffer,
            mt_buffer,
            fallback_mask,
            seq_map_r2c.npu(),
            state_v_first=state_v_first,
        )
    else:
        out = correct_initial_states(
            raw_h0,
            ht_buffer,
            mt_buffer,
            fallback_mask,
            seq_map_r2c.npu(),
            state_v_first=state_v_first,
            reverse=reverse,
            transpose_m=transpose_m,
        )

    max_iter = max(num_iters_per_batch)
    torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)

    kind = "terminal" if terminal else "initial"
    tag = f"{kind} raw={use_raw_h0} svf={state_v_first} rev={reverse} tm={transpose_m} fb={fb_pattern} {dtype}"
    print(f"[PASS] {tag} (max_iter={max_iter}, atol={atol:.4f})")

    for _ in range(10):
        if terminal:
            out = correct_terminal_states(
                raw_h0,
                ht_buffer,
                mt_buffer,
                fallback_mask,
                seq_map_r2c.npu(),
                state_v_first=state_v_first,
            )
        else:
            out = correct_initial_states(
                raw_h0,
                ht_buffer,
                mt_buffer,
                fallback_mask,
                seq_map_r2c.npu(),
                state_v_first=state_v_first,
                reverse=reverse,
                transpose_m=transpose_m,
            )


def main():
    fp16 = (torch.float16, 1e-2, 1e-2)
    fp32_out = (torch.float16, 1e-3, 1e-3)

    _run_case(True, False, False, False, "all_true", *fp16)
    _run_case(True, False, False, False, "all_false", *fp16)
    _run_case(True, False, False, False, "mixed", *fp16)
    _run_case(False, False, False, False, "mixed", *fp32_out)
    _run_case(True, False, True, True, "mixed", *fp16)
    _run_case(True, True, False, False, "mixed", *fp16)
    _run_case(False, True, False, False, "mixed", *fp32_out)
    _run_case(True, True, True, True, "all_true", *fp16)

    _run_case(True, False, True, True, "mixed", *fp16, terminal=True)
    _run_case(False, False, True, True, "mixed", *fp32_out, terminal=True)
    _run_case(True, True, True, True, "mixed", *fp16, terminal=True)

    print("\n\033[92mAll tests passed!\033[0m")


if __name__ == "__main__":
    main()
