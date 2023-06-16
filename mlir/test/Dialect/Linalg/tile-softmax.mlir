// RUN: mlir-opt %s -test-transform-dialect-interpreter -canonicalize --split-input-file | FileCheck %s


func.func @softmax(%arg0: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
  %0 = tensor.empty() : tensor<16x64x256xf32>
  %1 = linalg.softmax
         dimension(1) ins(%arg0 : tensor<16x64x256xf32>) outs(%0 : tensor<16x64x256xf32>) -> tensor<16x64x256xf32>
  return %1 : tensor<16x64x256xf32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 30)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
// CHECK:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
// CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index 
// CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:        %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG:        %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:        %[[C30:.+]] = arith.constant 30 : index
// CHECK-DAG:        %[[D0:.+]] = tensor.empty() : tensor<16x64x256xf32>
// CHECK-DAG:        %[[D1:.+]] = iree_input.dispatch.workgroup.id[0] : index
// CHECK-DAG:        %[[D2:.+]] = iree_input.dispatch.workgroup.count[0] : index
// CHECK-DAG:        %[[D3:.+]] = iree_input.dispatch.workgroup.id[1] : index
// CHECK-DAG:        %[[D4:.+]] = iree_input.dispatch.workgroup.count[1] : index
// CHECK-DAG:      %[[D5:.+]] = affine.apply #[[MAP]]()[%[[D3]]]
// CHECK-DAG:      %[[D6:.+]] = affine.apply #[[MAP]]()[%[[D4]]]
// CHECK:        %[[D7:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[D5]] to %[[C16]] step %[[D6]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<16x64x256xf32>) {
// CHECK-DAG:        %[[D8:.+]] = affine.min #[[MAP1]](%[[ARG1]])[%[[C10]], %[[C16]]]
// CHECK-DAG:        %[[D9:.+]] = affine.apply #[[MAP2]]()[%[[D1]]] 
// CHECK-DAG:        %[[D10:.+]] = affine.apply #[[MAP2]]()[%[[D2]]]
// CHECK:          %[[D11:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[D9]] to %[[C256]] step %[[D10]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<16x64x256xf32>) {
// CHECK-DAG:          %[[D12:.+]] = affine.min #[[MAP3]](%[[ARG3]])[%[[C30]], %[[C256]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, %[[ARG3]]] [%[[D8]],
// CHECK-SAME:         %[[C64]], %[[D12]]] [1, 1, 1] : tensor<16x64x256xf32> to tensor<?x?x?xf32>
// CHECK:            %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[D0]][%[[ARG1]], 0, %[[ARG3]]] [%[[D8]],
// CHECK-SAME:         %[[C64]], %[[D12]]] [1, 1, 1] : tensor<16x64x256xf32> to tensor<?x?x?xf32>
// CHECK:            %[[D13:.+]] = iree_linalg_ext.softmax {__internal_linalg_transform__ = "distribute_output"}
// CHECK-SAME:         dimension(1) ins(%[[EXTRACTED_SLICE]] : tensor<?x?x?xf32>) outs(%[[EXTRACTED_SLICE_0]] :
// CHECK-SAME:         tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D13]] into %[[ARG4]][%[[ARG1]], 0, %[[ARG3]]]
// CHECK-SAME:         [%[[D8]], %[[C64]], %[[D12]]] [1, 1, 1] : tensor<?x?x?xf32> into tensor<16x64x256xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<16x64x256xf32>
// CHECK:          } 
// CHECK:          scf.yield %[[D11]] : tensor<16x64x256xf32>
// CHECK:        } 
// CHECK:        return %[[D7]] : tensor<16x64x256xf32>
// CHECK:      } 

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile %0 [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

// -----

func.func @softmax_memref(%arg0: memref<16x64x256xf32>, %arg1: memref<16x64x256xf32>) {
  linalg.softmax 
    dimension(1) ins(%arg0 : memref<16x64x256xf32>) outs(%arg1 : memref<16x64x256xf32>)
  return
} 
// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (10, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 30)> 
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0)[s0, s1] -> (30, -d0 + s1)>
// CHECK:      func.func @softmax_memref(%[[ARG0:[a-zA-Z0-9_]+]]: memref<16x64x256xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   memref<16x64x256xf32>) { 
// CHECK-DAG:    %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:    %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:    %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG:    %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:    %[[C30:.+]] = arith.constant 30 : index
// CHECK:        %[[D0:.+]] = iree_input.dispatch.workgroup.id[0] : index
// CHECK:        %[[D1:.+]] = iree_input.dispatch.workgroup.count[0] : index
// CHECK:        %[[D2:.+]] = iree_input.dispatch.workgroup.id[1] : index
// CHECK:        %[[D3:.+]] = iree_input.dispatch.workgroup.count[1] : index
// CHECK-DAG:      %[[D4:.+]] = affine.apply #[[MAP]]()[%[[D2]]]
// CHECK-DAG:      %[[D5:.+]] = affine.apply #[[MAP]]()[%[[D3]]]
// CHECK:        scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[D4]] to %[[C16]] step %[[D5]] {
// CHECK-DAG:        %[[D6:.+]] = affine.min #[[MAP1]](%[[ARG2]])[%[[C10]], %[[C16]]]
// CHECK-DAG:        %[[D7:.+]] = affine.apply #[[MAP2]]()[%[[D0]]]
// CHECK-DAG:        %[[D8:.+]] = affine.apply #[[MAP2]]()[%[[D1]]]
// CHECK:          scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[D7]] to %[[C256]] step %[[D8]] {
// CHECK-DAG:          %[[D9:.+]] = affine.min #[[MAP3]](%[[ARG3]])[%[[C30]], %[[C256]]]
// CHECK:            %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[ARG2]], 0, %[[ARG3]]] [%[[D6]], %[[C64]], %[[D9]]]
// CHECK-SAME:         [1, 1, 1] : memref<16x64x256xf32> to memref<?x?x?xf32, strided<[16384, 256, 1], offset: ?>>
// CHECK:            %[[SUBVIEW_0:.+]] = memref.subview %[[ARG1]][%[[ARG2]], 0, %[[ARG3]]] [%[[D6]], %[[C64]], %[[D9]]]
// CHECK-SAME:         [1, 1, 1] : memref<16x64x256xf32> to memref<?x?x?xf32, strided<[16384, 256, 1], offset: ?>>
// CHECK:            iree_linalg_ext.softmax {__internal_linalg_transform__ = "distribute_output"} dimension(1)
// CHECK-SAME:         ins(%[[SUBVIEW]] : memref<?x?x?xf32, strided<[16384, 256, 1], offset: ?>>) outs(%[[SUBVIEW_0]] :
// CHECK-SAME:         memref<?x?x?xf32, strided<[16384, 256, 1], offset: ?>>)
// CHECK:          }      
// CHECK:        }
// CHECK:        return
// CHECK:      }



transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile %0 [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

