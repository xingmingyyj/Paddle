// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pir/transforms/gpu/fused_matmul_swiglu_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class FuseMatmulSwigluPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FuseMatmulSwigluPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    const auto &matmul_1 = src.Op(paddle::dialect::MatmulOp::name());
    src.Tensor("b") = matmul_1(src.Tensor("a"), src.Tensor("w1"));

    const auto &matmul_2 = src.Op(paddle::dialect::MatmulOp::name());
    src.Tensor("c") = matmul_2(src.Tensor("a"), src.Tensor("w2"));

    const auto &swiglu = src.Op(paddle::dialect::SwigluOp::name());
    src.Tensor("d") = swiglu(src.Tensor("b"), src.Tensor("c"));

    paddle::drr::ResultPattern res = src.ResultPattern();

    auto &combine_op = res.Op(pir::CombineOp::name());
    combine_op({&res.Tensor("w1"), &res.Tensor("w2")},
               {&res.Tensor("combine_out")});

    const auto &dtype_attr =
        res.ComputeAttr([](const paddle::drr::MatchContext &match_ctx) -> bool {
          return pir::GetDataTypeFromValue(match_ctx.Tensor("w1"));
        });

    auto &coalesce_tensor = res.Op(paddle::dialect::CoalesceTensorOp::name(),
                                   {{"dytpe", dtype_attr}});

    coalesce_tensor({&res.Tensor("combine_out")},
                    {&res.Tensor("combine_out"), &res.Tensor("fused_w")});
    auto &matmul = res.Op(paddle::dialect::MatmulOp::name());
    matmul({&res.Tensor("a"), &res.Tensor("fused_w")}, {&res.Tensor("b_c")});
    auto &split = res.Op(
        paddle::dialect::SplitOp::name(),
        {{"sections", res.VectorInt32Attr({2})}, {"axis", res.Int32Attr(-1)}});

    split({&res.Tensor("b_c")}, {&res.Tensor("b"), &res.Tensor("c")});

    auto &swiglu1 = res.Op(paddle::dialect::SwigluOp::name());
    swiglu1(res.Tensor("b_c"), res.Tensor("d"));
  }
};

class FuseMatmulSwigluPass : public pir::PatternRewritePass {
 public:
  FuseMatmulSwigluPass()
      : pir::PatternRewritePass("fuse_matmul_swiglu_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FuseMatmulSwigluPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedMatmulSwigluPass() {
  return std::make_unique<FuseMatmulSwigluPass>();
}

}  // namespace pir

REGISTER_IR_PASS(fuse_matmul_swiglu_pass, FuseMatmulSwigluPass);
