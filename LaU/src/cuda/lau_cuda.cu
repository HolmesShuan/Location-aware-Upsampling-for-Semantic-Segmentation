#include <vector>
#include "cuda/lau_cuda_kernel.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <THC/THC.h>
// #include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

// extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

at::Tensor
location_aware_upsampling_cuda_forward(const at::Tensor &input,
                    const at::Tensor &offset_x, 
                    const at::Tensor &offset_y, 
                    const int k_h,
                    const int k_w)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(offset_x.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(offset_y.type().is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);    // N
    const int channels = input.size(1); // C
    const int height = input.size(2);   // H
    const int width = input.size(3);    // W

    // printf("Kernels: %d %d %d %d\n", kernel_h_, kernel_w_, kernel_w, kernel_h);
    // printf("Channels: %d %d\n", channels, channels_kernel);
    // printf("Channels: %d %d\n", channels_out, channels_kernel);
    
    const int channels_out = channels;
    const int height_out = height * k_h;
    const int width_out = width * k_w;

    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    // define alias for easy use
    const int per_input_size = channels * height * width;
    const int per_output_size = channels_out * height_out * width_out;
    const int per_offset_size = offset_x.size(1) * offset_x.size(2) * offset_x.size(3);
    
    AT_ASSERTM(offset_x.size(1) == channels_out, "%d channel number mismatch.", channels_out);
    AT_ASSERTM(offset_x.size(2) == height_out, "%d height mismatch.", height_out);
    AT_ASSERTM(offset_x.size(3) == width_out, "%d width mismatch.", width_out);
    
    for (int n = 0; n < batch; ++n) {
        AT_DISPATCH_FLOATING_TYPES(input.type(), "location_aware_upsampling_forward_cuda", ([&] {
            lau_bilinear_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * per_input_size,
                                             offset_x.data<scalar_t>() + n * per_offset_size,
                                             offset_y.data<scalar_t>() + n * per_offset_size,
                                             1, channels, height, width,
                                             height_out, width_out,
                                             output.data<scalar_t>() + n * per_output_size);

        }));
    }

    output = output.contiguous();

    return output;
}

std::vector<at::Tensor> location_aware_upsampling_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &offset_x,
                                             const at::Tensor &offset_y,
                                             const at::Tensor &grad_output)
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset_x.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(offset_y.type().is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int batch_ = grad_output.size(0);
    const int channels_out_ = grad_output.size(1);
    const int height_out_ = grad_output.size(2);
    const int width_out_ = grad_output.size(3);

    auto grad_input = at::zeros_like(input);
    auto grad_offset_x = at::zeros_like(offset_x);
    auto grad_offset_y = at::zeros_like(offset_y);

    const int per_input_size = channels * height * width;
    const int per_output_size = channels_out_ * height_out_ * width_out_;
    const int per_offset_size = offset_x.size(1) * offset_x.size(2) * offset_x.size(3);

    for (int n = 0; n < batch; ++n) {
        AT_DISPATCH_FLOATING_TYPES(input.type(), "location_aware_upsampling_backward_cuda", ([&] {
            // gradient w.r.t. input data and coords
            lau_bilinear_cuda_backward(at::cuda::getCurrentCUDAStream(),
                                                   grad_output.data<scalar_t>() + n * per_output_size,
                                                   input.data<scalar_t>() + n * per_input_size,
                                                   offset_x.data<scalar_t>() + n * per_offset_size,
                                                   offset_y.data<scalar_t>() + n * per_offset_size,
                                                   1, channels, height, width,
                                                   height_out_, width_out_,
                                                   grad_offset_x.data<scalar_t>() + n * per_offset_size,
                                                   grad_offset_y.data<scalar_t>() + n * per_offset_size,
                                                   grad_input.data<scalar_t>() + n * per_input_size);
        }));
    }

    return {
        grad_input, grad_offset_x, grad_offset_y
    };
}

at::Tensor
location_determined_upsampling_cuda_forward(const at::Tensor &input,
                    const int k_h,
                    const int k_w)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");

    const int batch = input.size(0);    // N
    const int channels = input.size(1); // C
    const int height = input.size(2);   // H
    const int width = input.size(3);    // W

    // printf("Kernels: %d %d %d %d\n", kernel_h_, kernel_w_, kernel_w, kernel_h);
    // printf("Channels: %d %d\n", channels, channels_kernel);
    // printf("Channels: %d %d\n", channels_out, channels_kernel);
    
    const int channels_out = channels;
    const int height_out = height * k_h;
    const int width_out = width * k_w;

    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    // define alias for easy use
    const int per_input_size = channels * height * width;
    const int per_output_size = channels_out * height_out * width_out;
    
    for (int n = 0; n < batch; ++n) {
        AT_DISPATCH_FLOATING_TYPES(input.type(), "location_determined_upsampling_forward_cuda", ([&] {
            ldu_bilinear_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * per_input_size,
                                             1, channels, height, width,
                                             height_out, width_out,
                                             output.data<scalar_t>() + n * per_output_size);
        }));
    }

    output = output.contiguous();

    return output;
}

at::Tensor 
location_determined_upsampling_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &grad_output)
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");

    auto grad_input = at::zeros_like(input);
    
    // NOT IMPLEMENTED

    return grad_input;
}

std::vector<at::Tensor>
location_determined_upsampling_multi_output_cuda_forward(const at::Tensor &input,
                    const int k_h,
                    const int k_w)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");

    const int batch = input.size(0);    // N
    const int channels = input.size(1); // C
    const int height = input.size(2);   // H
    const int width = input.size(3);    // W

    // printf("Kernels: %d %d %d %d\n", kernel_h_, kernel_w_, kernel_w, kernel_h);
    // printf("Channels: %d %d\n", channels, channels_kernel);
    // printf("Channels: %d %d\n", channels_out, channels_kernel);
    
    const int channels_out = channels;
    const int height_out = height * k_h;
    const int width_out = width * k_w;

    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());
    auto output_lt = at::empty({batch, channels_out, height_out, width_out}, input.options());
    auto output_lb = at::empty({batch, channels_out, height_out, width_out}, input.options());
    auto output_rt = at::empty({batch, channels_out, height_out, width_out}, input.options());
    auto output_rb = at::empty({batch, channels_out, height_out, width_out}, input.options());

    // define alias for easy use
    const int per_input_size = channels * height * width;
    const int per_output_size = channels_out * height_out * width_out;
    
    for (int n = 0; n < batch; ++n) {
        AT_DISPATCH_FLOATING_TYPES(input.type(), "location_determined_upsampling_forward_cuda", ([&] {
            ldu_bilinear_multi_output_cuda(at::cuda::getCurrentCUDAStream(),
                                             input.data<scalar_t>() + n * per_input_size,
                                             1, channels, height, width,
                                             height_out, width_out,
                                             output.data<scalar_t>() + n * per_output_size,
                                             output_lt.data<scalar_t>() + n * per_output_size,
                                             output_lb.data<scalar_t>() + n * per_output_size,
                                             output_rt.data<scalar_t>() + n * per_output_size,
                                             output_rb.data<scalar_t>() + n * per_output_size);
        }));
    }

    output = output.contiguous();
    output_lt = output_lt.contiguous();
    output_lb = output_lb.contiguous();
    output_rt = output_rt.contiguous();
    output_rb = output_rb.contiguous();

    return {output, output_lt, output_lb, output_rt, output_rb};
}

at::Tensor 
location_determined_upsampling_multi_output_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &grad_output)
{

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");

    auto grad_input = at::zeros_like(input);
    
    // NOT IMPLEMENTED

    return grad_input;
}