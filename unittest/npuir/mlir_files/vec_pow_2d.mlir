module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [128, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [128, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [128, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %alloc = memref.alloc() : memref<32x64xf16, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<32x64xf16, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<32x64xf16, #hivm.address_space<ub>>
    %2 = arith.divsi %1, %c2_i32 : i32
    %3 = arith.muli %2, %c32_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.remsi %1, %c2_i32 : i32
    %6 = arith.muli %5, %c64_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %subview = memref.subview %reinterpret_cast[%4, %7] [32, 64] [1, 1] : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc : memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x64xf16, #hivm.address_space<ub>>
    %subview_4 = memref.subview %reinterpret_cast_1[%4, %7] [32, 64] [1, 1] : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_4, %alloc_2 : memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x64xf16, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<32x64xf16>
    memref.copy %alloc, %alloc_5 : memref<32x64xf16, #hivm.address_space<ub>> to memref<32x64xf16>
    %8 = bufferization.to_tensor %alloc_5 restrict : memref<32x64xf16>
    %alloc_6 = memref.alloc() : memref<32x64xf16>
    memref.copy %alloc_2, %alloc_6 : memref<32x64xf16, #hivm.address_space<ub>> to memref<32x64xf16>
    %9 = bufferization.to_tensor %alloc_6 restrict : memref<32x64xf16>
    %10 = tensor.empty() : tensor<32x64xf16>
    %11 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%8, %9 : tensor<32x64xf16>, tensor<32x64xf16>) outs(%10 : tensor<32x64xf16>) -> tensor<32x64xf16>
    bufferization.materialize_in_destination %11 in writable %alloc_3 : (tensor<32x64xf16>, memref<32x64xf16, #hivm.address_space<ub>>) -> ()
    %subview_7 = memref.subview %reinterpret_cast_0[%4, %7] [32, 64] [1, 1] : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc_3, %subview_7 : memref<32x64xf16, #hivm.address_space<ub>> to memref<32x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
