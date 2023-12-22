/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/meta_tensor.h"

namespace phi {

// Common InferMeta Functions for binary operators, The format like:
//
//   1. void [FunctionDesc|OpName]InferMeta(const MetaTensor& x,
//                                          const MetaTensor& y,
//                                          ...,
//                                          MetaTensor* out) {}
//
// NOTE: The name "InferShape" may be not appropriate. "InferMeta" may be good.
//   Because functions in this file not only can infer shape, but also need
//   infer lod or other useful data.
//
// The InferMeta Functions in this file are arranged in alphabetic order.

void AllValueCompareInferMeta(const MetaTensor& x,
                              const MetaTensor& y,
                              MetaTensor* out,
                              MetaConfig config = MetaConfig());

void KLDivInferMeta(const MetaTensor& x,
                    const MetaTensor& label,
                    const std::string& reduction,
                    MetaTensor* out,
                    MetaConfig config = MetaConfig());

void ArrayWriteInferMeta(const MetaTensor& array,
                         const MetaTensor& x,
                         MetaTensor* out,
                         MetaConfig config = MetaConfig());

void ArrayReadInferMeta(const MetaTensor& array,
                        const Scalar& i,
                        MetaTensor* out,
                        MetaConfig config = MetaConfig());

void Atan2InferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out);

void BCELossInferMeta(const MetaTensor& input,
                      const MetaTensor& label,
                      MetaTensor* out,
                      MetaConfig config = MetaConfig());

void BincountInferMeta(const MetaTensor& x,
                       const MetaTensor& weights,
                       const Scalar& minlength,
                       MetaTensor* out);

void BinomialInferMeta(const MetaTensor& count,
                       const MetaTensor& prob,
                       MetaTensor* out,
                       MetaConfig config = MetaConfig());

void BmmInferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out);

void CholeskySolveInferMeta(const MetaTensor& x,
                            const MetaTensor& y,
                            bool upper,
                            MetaTensor* out);

void CompareAllInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         MetaTensor* out);

void CompareInferMeta(const MetaTensor& x,
                      const MetaTensor& y,
                      MetaTensor* out);

void CompareRawInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         int axis,
                         MetaTensor* out);

void ComplexInferMeta(const MetaTensor& x,
                      const MetaTensor& y,
                      MetaTensor* out);

void ConvInferMeta(const MetaTensor& input,
                   const MetaTensor& filter,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::string& padding_algorithm,
                   const std::vector<int>& dilations,
                   int groups,
                   const std::string& data_format,
                   MetaTensor* out,
                   MetaConfig config = MetaConfig());

void Conv3DInferMeta(const MetaTensor& input,
                     const MetaTensor& filter,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::string& padding_algorithm,
                     int groups,
                     const std::vector<int>& dilations,
                     const std::string& data_format,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void ConvTransposeInferMeta(const MetaTensor& x,
                            const MetaTensor& filter,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& output_padding,
                            const std::vector<int>& output_size,
                            const std::string& padding_algorithm,
                            int groups,
                            const std::vector<int>& dilations,
                            const std::string& data_format,
                            MetaTensor* out,
                            MetaConfig config = MetaConfig());

void Conv2dTransposeInferMeta(const MetaTensor& x,
                              const MetaTensor& filter,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::vector<int>& output_padding,
                              const IntArray& output_size,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::vector<int>& dilations,
                              const std::string& data_format,
                              MetaTensor* out,
                              MetaConfig config = MetaConfig());

void CrossInferMeta(const MetaTensor& x,
                    const MetaTensor& y,
                    int axis,
                    MetaTensor* out);

void CrossEntropyWithSoftmaxInferMeta(const MetaTensor& logits,
                                      const MetaTensor& label,
                                      bool soft_label,
                                      bool use_softmax,
                                      bool numeric_stable_mode,
                                      int ignore_index,
                                      int axis,
                                      MetaTensor* softmax,
                                      MetaTensor* loss,
                                      MetaConfig config = MetaConfig());

void DepthwiseConvInferMeta(const MetaTensor& input,
                            const MetaTensor& filter,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::string& padding_algorithm,
                            int groups,
                            const std::vector<int>& dilations,
                            const std::string& data_format,
                            MetaTensor* out,
                            MetaConfig config = MetaConfig());

void DistInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   float p,
                   MetaTensor* out);

void DistributeFpnProposalsInferMeta(
    const MetaTensor& fpn_rois,
    const MetaTensor& rois_num,
    int min_level,
    int max_level,
    int refer_level,
    int refer_scale,
    bool pixel_offset,
    std::vector<MetaTensor*> multi_fpn_rois,
    std::vector<MetaTensor*> multi_level_rois_num,
    MetaTensor* restore_index,
    MetaConfig config = MetaConfig());

void DotInferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out);

void DropoutInferMeta(const MetaTensor& x,
                      const MetaTensor& seed_tensor,
                      const Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      MetaTensor* out,
                      MetaTensor* mask);

void DropoutNdInferMeta(const MetaTensor& x,
                        const MetaTensor& seed_tensor,
                        const Scalar& p,
                        bool is_test,
                        const std::string& mode,
                        int seed,
                        bool fix_seed,
                        const std::vector<int>& axis,
                        MetaTensor* out,
                        MetaTensor* mask);

void ElementwiseInferMeta(const MetaTensor& x,
                          const MetaTensor& y,
                          MetaTensor* out);

void ElementwiseRawInferMeta(const MetaTensor& x_meta,
                             const MetaTensor& y_meta,
                             int axis,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

void EmbeddingInferMeta(const MetaTensor& x,
                        const MetaTensor& weight,
                        int64_t padding_idx,
                        MetaTensor* out);

void CEmbeddingInferMeta(const MetaTensor& weight,
                         const MetaTensor& x,
                         int64_t start_index,
                         MetaTensor* out);

void ExpandAsInferMeta(const MetaTensor& x,
                       const MetaTensor& y,
                       const std::vector<int>& target_shape,
                       MetaTensor* out);

void FillDiagonalTensorInferMeta(const MetaTensor& x,
                                 const MetaTensor& y,
                                 int64_t offset,
                                 int dim1,
                                 int dim2,
                                 MetaTensor* out);

void FusedDropoutAddInferMeta(const MetaTensor& x,
                              const MetaTensor& y,
                              MetaTensor* out,
                              MetaTensor* seed_offset);

void FusedMatmulInferMeta(const MetaTensor& x,
                          const MetaTensor& y,
                          const MetaTensor& residual_data,
                          bool transpose_x,
                          bool transpose_y,
                          const float matmul_alpha,
                          const std::string& fuse_activation,
                          const float fuse_lapha,
                          const float fuse_beat,
                          const float fused_output_scale,
                          const std::vector<int>& fused_reshape_X,
                          const std::vector<int>& fused_transpose_X,
                          const std::vector<int>& fused_reshape_Y,
                          const std::vector<int>& fused_transpose_Y,
                          const std::vector<int>& fused_reshape_Out,
                          const std::vector<int>& fused_transpose_Out,
                          const std::string& mkldnn_data_type,
                          const float scale_x,
                          const float scale_y,
                          const float scale_scale_in_eltwise,
                          const float scale_out,
                          const bool force_fp32_output,
                          MetaTensor* out);

void GatherInferMeta(const MetaTensor& x,
                     const MetaTensor& index,
                     const Scalar& axis,
                     MetaTensor* out);

void GatherNdInferMeta(const MetaTensor& x,
                       const MetaTensor& index,
                       MetaTensor* out);

void GatherTreeMeta(const MetaTensor& ids,
                    const MetaTensor& parents,
                    MetaTensor* out);

void GridSampleBaseInferMeta(const MetaTensor& x,
                             const MetaTensor& grid,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

void HuberLossInferMeta(const MetaTensor& input_meta,
                        const MetaTensor& label_meta,
                        float delta,
                        MetaTensor* out,
                        MetaTensor* residual,
                        MetaConfig config = MetaConfig());

void IdentityLossGradInferMeta(const MetaTensor& x,
                               const MetaTensor& out_grad,
                               const int reduction,
                               MetaTensor* x_grad);

void IndexSampleInferMeta(const MetaTensor& x,
                          const MetaTensor& y,
                          MetaTensor* out,
                          MetaConfig config = MetaConfig());

void IndexSelectInferMeta(const MetaTensor& x,
                          const MetaTensor& index,
                          int dim,
                          MetaTensor* output);

void IndexSelectStridedInferMeta(const MetaTensor& x,
                                 int64_t index,
                                 int dim,
                                 MetaTensor* output);

void IndexAddInferMeta(const MetaTensor& x,
                       const MetaTensor& index,
                       const MetaTensor& add_value,
                       int axis,
                       MetaTensor* output);

void KronInferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out);

void LogicalBinaryInferMeta(const MetaTensor& x,
                            const MetaTensor& y,
                            MetaTensor* out);

void LogLossInferMeta(const MetaTensor& input,
                      const MetaTensor& label,
                      float epsilon,
                      MetaTensor* out,
                      MetaConfig config = MetaConfig());

void LUUnpackInferMeta(const MetaTensor& x,
                       const MetaTensor& pivots,
                       bool unpack_ludata,
                       bool unpack_pivots,
                       MetaTensor* pmat,
                       MetaTensor* l,
                       MetaTensor* u);

void MarginCrossEntropyInferMeta(const MetaTensor& logits,
                                 const MetaTensor& label,
                                 bool return_softmax,
                                 int ring_id,
                                 int rank,
                                 int nranks,
                                 float margin1,
                                 float margin2,
                                 float margin3,
                                 float scale,
                                 MetaTensor* softmax,
                                 MetaTensor* loss,
                                 MetaConfig config = MetaConfig());

void MaskedSelectInferMeta(const MetaTensor& x,
                           const MetaTensor& mask,
                           MetaTensor* out);

void MatmulInferMeta(const MetaTensor& x,
                     const MetaTensor& y,
                     bool trans_x,
                     bool trans_y,
                     MetaTensor* out);

void MatmulWithFlattenInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                int x_num_col_dims,
                                int y_num_col_dims,
                                MetaTensor* out);

void MatrixNMSInferMeta(const MetaTensor& bboxes,
                        const MetaTensor& scores,
                        float score_threshold,
                        int nms_top_k,
                        int keep_top_k,
                        float post_threshold,
                        bool use_gaussian,
                        float gaussian_sigma,
                        int background_label,
                        bool normalized,
                        MetaTensor* out,
                        MetaTensor* index,
                        MetaTensor* roisnum,
                        MetaConfig config = MetaConfig());

void MatrixRankStaticInferMeta(const MetaTensor& x,
                               const MetaTensor& atol_tensor,
                               bool use_default_tol,
                               bool hermitian,
                               MetaTensor* out);

void MatrixRankTolInferMeta(const MetaTensor& x,
                            const MetaTensor& atol_tensor,
                            bool use_default_tol,
                            bool hermitian,
                            MetaTensor* out);

void MvInferMeta(const MetaTensor& x, const MetaTensor& vec, MetaTensor* out);

void PReluInferMeta(const MetaTensor& x,
                    const MetaTensor& alpha,
                    const std::string& data_format,
                    const std::string& mode,
                    MetaTensor* out,
                    MetaConfig config = MetaConfig());

void RepeatInterleaveWithTensorIndexInferMeta(const MetaTensor& x,
                                              const MetaTensor& repeats,
                                              int dim,
                                              MetaTensor* out);

void RowConvInferMeta(const MetaTensor& x,
                      const MetaTensor& filter,
                      MetaTensor* out);

void PriorBoxInferMeta(const MetaTensor& input,
                       const MetaTensor& image,
                       const std::vector<float>& min_sizes,
                       const std::vector<float>& max_sizes,
                       const std::vector<float>& aspect_ratios,
                       const std::vector<float>& variances,
                       bool flip,
                       bool clip,
                       float step_w,
                       float step_h,
                       float offset,
                       bool min_max_aspect_ratios_order,
                       MetaTensor* out,
                       MetaTensor* var);

void SearchsortedInferMeta(const MetaTensor& sorted_sequence,
                           const MetaTensor& value,
                           bool out_int32,
                           bool right,
                           MetaTensor* out);

void SequenceMaskInferMeta(const MetaTensor& x,
                           const MetaTensor& max_len_tensor,
                           int maxlen,
                           int out_dtype,
                           MetaTensor* y);

void SoftmaxMaskFuseInferMeta(const MetaTensor& x,
                              const MetaTensor& mask,
                              MetaTensor* out);

void SegmentPoolInferMeta(const MetaTensor& x,
                          const MetaTensor& segment_ids,
                          const std::string& pooltype,
                          MetaTensor* out,
                          MetaTensor* summed_ids,
                          MetaConfig config = MetaConfig());

void TakeAlongAxisInferMeta(const MetaTensor& x,
                            const MetaTensor& index,
                            int axis,
                            MetaTensor* out);

void TriangularSolveInferMeta(const MetaTensor& x,
                              const MetaTensor& y,
                              bool upper,
                              bool transpose,
                              bool unitriangular,
                              MetaTensor* out);

void TopPSamplingInferMeta(const MetaTensor& x,
                           const MetaTensor& ps,
                           const MetaTensor& threshold,
                           int random_seed,
                           MetaTensor* out,
                           MetaTensor* ids);

void LstsqInferMeta(const MetaTensor& x,
                    const MetaTensor& y,
                    const Scalar& rcond,
                    const std::string& driver,
                    MetaTensor* solution,
                    MetaTensor* residuals,
                    MetaTensor* rank,
                    MetaTensor* singular_values);

void YoloBoxInferMeta(const MetaTensor& x,
                      const MetaTensor& img_size,
                      const std::vector<int>& anchors,
                      int class_num,
                      float conf_thresh,
                      int downsample_ratio,
                      bool clip_bbox,
                      float scale_x_y,
                      bool iou_aware,
                      float iou_aware_factor,
                      MetaTensor* boxes,
                      MetaTensor* scores,
                      MetaConfig config = MetaConfig());

void ValueCompareInferMeta(const MetaTensor& x,
                           const MetaTensor& y,
                           MetaTensor* out,
                           MetaConfig config = MetaConfig());

void SolveInferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out);

void UnpoolInferMeta(const MetaTensor& x,
                     const MetaTensor& indices,
                     const std::vector<int>& ksize,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const IntArray& output_size,
                     const std::string& data_format,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void Unpool3dInferMeta(const MetaTensor& x,
                       const MetaTensor& indices,
                       const std::vector<int>& ksize,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::vector<int>& output_size,
                       const std::string& data_format,
                       MetaTensor* out,
                       MetaConfig config = MetaConfig());

void WeightDequantizeInferMeta(const MetaTensor& x,
                               const MetaTensor& scale,
                               const std::string& algo,
                               DataType out_dtype,
                               MetaTensor* out);

void LookupTableInferMeta(const MetaTensor& w,
                          const MetaTensor& ids,
                          MetaTensor* out

);

}  // namespace phi
