#pragma once

#ifdef WITH_CUDA
#include "cuda/lau_cuda.h"
#endif

at::Tensor
location_aware_upsampling_forward(const at::Tensor &input,
               const at::Tensor &offset_x,
               const at::Tensor &offset_y,
               const int k_h,
               const int k_w)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return location_aware_upsampling_cuda_forward(input,
                                    offset_x, 
                                    offset_y,
                                    k_h, 
                                    k_w);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
location_aware_upsampling_backward(const at::Tensor &input,
                     const at::Tensor &offset_x,
                     const at::Tensor &offset_y,
                     const at::Tensor &grad_output)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return location_aware_upsampling_cuda_backward(input,
                                    offset_x,
                                    offset_y,
                                    grad_output);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor
location_determined_upsampling_forward(const at::Tensor &input,
               const int k_h,
               const int k_w)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return location_determined_upsampling_cuda_forward(input,
                                    k_h, 
                                    k_w);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor
location_determined_upsampling_backward(const at::Tensor &input,
                     const at::Tensor &offset_x,
                     const at::Tensor &offset_y,
                     const at::Tensor &grad_output)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return location_determined_upsampling_cuda_backward(input,
                                    grad_output);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
location_determined_upsampling_multi_output_forward(const at::Tensor &input,
               const int k_h,
               const int k_w)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return location_determined_upsampling_multi_output_cuda_forward(input,
                                    k_h, 
                                    k_w);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor
location_determined_upsampling_multi_output_backward(const at::Tensor &input,
                     const at::Tensor &offset_x,
                     const at::Tensor &offset_y,
                     const at::Tensor &grad_output)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return location_determined_upsampling_multi_output_cuda_backward(input,
                                    grad_output);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}