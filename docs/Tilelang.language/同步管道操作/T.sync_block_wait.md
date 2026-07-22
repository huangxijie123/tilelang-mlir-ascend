# Tilelang.language.sync_block_wait

## 1. OP概述

简介：`tilelang.language.sync_block_wait`用于Block内部的同步。让当前的执行处于等待状态，直到指定的事件标志位被对应的sync_block_set指令激活

```
T.sync_block_wait(id)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `id` | `int` | 同步标志位ID（Flag ID） |

### 2.2 支持规格

#### 2.2.1 DataType支持

不涉及

#### 2.2.2 Shape支持

不涉及

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下代码示例了sync_block_wait同步指令的使用

```python
def simple_sync_demo(M, N, K, block_M, block_N, dtype="float16", inner_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, subid):
            with T.Scope("Cube"):
                bx = (cid // n_num) * block_M
                by = (cid % n_num) * block_N

                A_BUF = T.alloc_L1((block_M, K), dtype)
                B_BUF = T.alloc_L1((K, block_N), dtype)

                T.load_nd2nz(A[bx, 0], A_BUF, [block_M, K])
                T.load_nd2nz(B[0, by], B_BUF, [K, block_N])

                C_BUF = T.alloc_L0C((block_M, block_N), inner_dtype)
                T.gemm(A_BUF, B_BUF, C_BUF, [block_M, K, block_N], initC=True)

                with T.rs("PIPE_FIX"):
                    T.sync_block_wait(1)
                    T.store_fixpipe(C_BUF, C[bx, by], [block_M, block_N], enable_nz2nd=True)
                    T.sync_block_set(0)

            with T.Scope("Vector"):
                bx = (cid // n_num) * block_M
                by = (cid % n_num) * block_N

                C_VEC = T.alloc_ub((block_M, block_N), dtype)
                D_VEC = T.alloc_ub((block_M, block_N), dtype)

                with T.rs("PIPE_MTE2"):
                    T.sync_block_set(1)
                    T.sync_block_wait(0)
                    T.copy(C[bx, by], C_VEC)

                T.vexp(C_VEC, D_VEC)
                T.copy(D_VEC, D[bx, by])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::sync_block_waitOp**将被编译为hivm::SyncBlockWaitOp
