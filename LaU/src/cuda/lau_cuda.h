#pragma once
#include <torch/extension.h>

at::Tensor
location_aware_upsampling_cuda_forward(const at::Tensor &input,
                    const at::Tensor &offset_x,
                    const at::Tensor &offset_y,
                    const int k_h,
                    const int k_w);

std::vector<at::Tensor>
location_aware_upsampling_cuda_backward(const at::Tensor &input,
                     const at::Tensor &offset_x,
                     const at::Tensor &offset_y,
                     const at::Tensor &grad_output);

at::Tensor
location_determined_upsampling_cuda_forward(const at::Tensor &input,
                    const int k_h,
                    const int k_w);

at::Tensor
location_determined_upsampling_cuda_backward(const at::Tensor &input,
                     const at::Tensor &grad_output);

std::vector<at::Tensor>
location_determined_upsampling_multi_output_cuda_forward(const at::Tensor &input,
                    const int k_h,
                    const int k_w);

at::Tensor
location_determined_upsampling_multi_output_cuda_backward(const at::Tensor &input,
                     const at::Tensor &grad_output);