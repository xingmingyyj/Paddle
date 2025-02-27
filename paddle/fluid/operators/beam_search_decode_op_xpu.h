/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/operators/beam_search_decode_op_def.h"
#include "paddle/phi/common/memory_utils.h"

namespace paddle {
namespace operators {

int SetMeta(const phi::DenseTensor& srcTensor, phi::DenseTensor* dstTensor) {
  if (srcTensor.dtype() == phi::DataType::INT32 ||
      srcTensor.dtype() == phi::DataType::INT64 ||
      srcTensor.dtype() == phi::DataType::FLOAT32 ||
      srcTensor.dtype() == phi::DataType::FLOAT16 ||
      srcTensor.dtype() == phi::DataType::FLOAT64) {
    const phi::DenseTensorMeta meta_data(srcTensor.dtype(), srcTensor.dims());
    dstTensor->set_meta(meta_data);
  } else {
    return xpu::Error_t::INVALID_PARAM;
  }

  return xpu::Error_t::SUCCESS;
}
template <typename T>
int CopyTensorByXPU(const phi::DenseTensor& srcTensor,
                    phi::DenseTensor* dstTensor,
                    int flag,
                    const Place& place) {
  const T* srcData = srcTensor.template data<T>();
  if (nullptr == srcData || nullptr == dstTensor || flag < 0 || flag > 1)
    return xpu::Error_t::INVALID_PARAM;

  int r = SetMeta(srcTensor, dstTensor);
  PADDLE_ENFORCE_EQ(
      r,
      xpu::Error_t::SUCCESS,
      common::errors::External("Execute function SetMeta failed by [%d]", r));

  if (flag == 0) {
    T* dstData = dstTensor->template mutable_data<T>(phi::CPUPlace());
    phi::memory_utils::Copy(phi::CPUPlace(),
                            dstData,
                            place,
                            srcData,
                            srcTensor.numel() * sizeof(T));
  } else {
    T* dstData = dstTensor->template mutable_data<T>(place);
    phi::memory_utils::Copy(place,
                            dstData,
                            phi::CPUPlace(),
                            srcData,
                            srcTensor.numel() * sizeof(T));
  }

  return xpu::Error_t::SUCCESS;
}

const int CopyTensorByType(const phi::DenseTensor& srcTensor,
                           phi::DenseTensor* dstTensor,
                           int flag,
                           const Place& place) {
  int r = 0;
  if (srcTensor.dtype() == phi::DataType::FLOAT32)
    r = CopyTensorByXPU<float>(srcTensor, dstTensor, flag, place);
  else if (srcTensor.dtype() == phi::DataType::FLOAT16)
    r = CopyTensorByXPU<phi::dtype::float16>(srcTensor, dstTensor, flag, place);
  else if (srcTensor.dtype() == phi::DataType::FLOAT64)
    r = CopyTensorByXPU<double>(srcTensor, dstTensor, flag, place);
  else if (srcTensor.dtype() == phi::DataType::INT32)
    r = CopyTensorByXPU<int>(srcTensor, dstTensor, flag, place);
  else if (srcTensor.dtype() == phi::DataType::INT64)
    r = CopyTensorByXPU<int64_t>(srcTensor, dstTensor, flag, place);
  else
    return xpu::Error_t::INVALID_PARAM;

  PADDLE_ENFORCE_EQ(r,
                    xpu::Error_t::SUCCESS,
                    common::errors::External(
                        "Execute function CopyTensorByXPU failed by [%d]", r));

  return xpu::Error_t::SUCCESS;
}

struct BeamSearchDecodeXPUFunctor {
  BeamSearchDecodeXPUFunctor(const LoDTensorArray& step_ids,
                             const LoDTensorArray& step_scores,
                             phi::DenseTensor* id_tensor,
                             phi::DenseTensor* score_tensor,
                             size_t beam_size,
                             int end_id)
      : beam_size_(beam_size),
        end_id_(end_id),
        id_tensor_(id_tensor),
        score_tensor_(score_tensor) {
    int r = 0;

    // First make a copy of XPU data on CPU
    if (step_ids[0].place().GetType() == phi::AllocationType::XPU) {
      // Copy all tensors in the input tensor array
      for (auto& step_id : step_ids) {
        phi::DenseTensor out;
        if (step_id.numel() > 0) {
          r = CopyTensorByType(step_id, &out, 0, step_ids[0].place());
          PADDLE_ENFORCE_EQ(
              r,
              xpu::Error_t::SUCCESS,
              common::errors::External(
                  "Execute function CopyTensorByXPU failed by [%d]", r));
        }

        out.set_lod(step_id.lod());
        step_ids_.push_back(out);
      }
    }

    if (step_scores[0].place().GetType() == phi::AllocationType::XPU) {
      // Copy all tensors in the input tensor array
      for (auto& step_score : step_scores) {
        phi::DenseTensor out;
        if (step_score.numel() > 0) {
          r = CopyTensorByType(step_score, &out, 0, step_scores[0].place());
          PADDLE_ENFORCE_EQ(
              r,
              xpu::Error_t::SUCCESS,
              common::errors::External(
                  "Execute function CopyTensorByType failed by [%d]", r));
        }

        out.set_lod(step_score.lod());
        step_scores_.push_back(out);
      }
    }
  }

  template <typename T>
  void apply_xpu() const {
    if (std::is_same<bool, T>::value) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "beam search decode op does not support bool!"));
    } else {
      BeamSearchDecoder<T> beam_search_decoder(beam_size_, end_id_);
      beam_search_decoder.Backtrace(
          step_ids_, step_scores_, id_tensor_, score_tensor_);
    }
  }

  size_t beam_size_;
  int end_id_;
  // TODO(Superjomn) Here might result serious performance issue in the
  // concurrency
  // scenarios.
  LoDTensorArray step_ids_ = LoDTensorArray();
  LoDTensorArray step_scores_ = LoDTensorArray();
  phi::DenseTensor* id_tensor_;
  phi::DenseTensor* score_tensor_;
};

}  // namespace operators
};  // namespace paddle
