//===- TilingInterface.td - Interface for tiling operations *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an interface to allow operations to generate a sequence of
// simpler ops of themselves.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AGGREGATEDOPINTERFACE
#define MLIR_AGGREGATEDOPINTERFACE

include "mlir/IR/OpBase.td"

def AggregatedOpInterface : OpInterface<"AggregatedOpInterface"> {
  let description = [{
    Interface for decomposing aggregated operations into a sequence of simpler
    ops.
  }];
  let cppNamespace = "::mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Method to decompose the operation into simpler operations.

          On success, this method returns the `Value`s that can be used
          to replace the results of the current operation.
          Therefore, it must return one value per result.
        }],
        /*retType=*/"FailureOr<SmallVector<Value>>",
        /*methodName=*/"decomposeOperation",
        /*args=*/(ins
            "OpBuilder &":$b),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return {};
        }]
      >
  ];
}
#endif // MLIR_AGGREGATEDOPINTERFACE
