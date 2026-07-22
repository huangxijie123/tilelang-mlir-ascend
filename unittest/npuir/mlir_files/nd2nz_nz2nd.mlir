module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @main_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [%c16, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [16, 16], strides: [%c16, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [%c16, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
    %alloc_2 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>) outs(%alloc : memref<16x16xf16, #hivm.address_space<cbuf>>)
    hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_1 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>) outs(%alloc_2 : memref<16x16xf16, #hivm.address_space<cbuf>>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16, #hivm.address_space<cbuf>>
    %1 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x16xf16, #hivm.address_space<cbuf>>
    %2 = bufferization.alloc_tensor() {memory_space = #hivm.address_space<cc>} : tensor<16x16xf32>
    %3 = hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%3 : tensor<16x16xf32>) outs(%reinterpret_cast_0 : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>)
    return
  }
  func.func @main_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    hivm.hir.set_ffts_base_addr %arg0
    return
  }
}
