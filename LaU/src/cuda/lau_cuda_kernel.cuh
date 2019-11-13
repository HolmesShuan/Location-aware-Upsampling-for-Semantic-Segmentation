#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__device__ scalar_t lau_micro_bilinear(const scalar_t *bottom_data, 
                                        const int data_width,
                                        const int height, 
                                        const int width, 
                                        scalar_t h, 
                                        scalar_t w)
{
  scalar_t upper_bound_x = height - 1.;
  scalar_t upper_bound_y = width - 1.;

  if (h > upper_bound_x)
      h = upper_bound_x;
  if (h < 0.) 
      h = 0.;
  if (w > upper_bound_y) 
      w = upper_bound_y;
  if (w < 0.) 
      w = 0.;
    
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = ceil(h);
  int w_high = ceil(w);
  
  scalar_t lt_rb_h_ne = 1.;
  scalar_t lt_rb_w_ne = 1.;
  
  if (h_low == h_high)
      lt_rb_h_ne = 0.;
  if (w_low == w_high)
      lt_rb_w_ne = 0.;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1. - lh, hw = 1. - lw;

  scalar_t v1 = bottom_data[h_low * data_width + w_low]; // left top
  scalar_t v2 = bottom_data[h_low * data_width + w_high]*lt_rb_w_ne; // right top
  scalar_t v3 = bottom_data[h_high * data_width + w_low]*lt_rb_h_ne; // left bottom
  scalar_t v4 = bottom_data[h_high * data_width + w_high]*lt_rb_h_ne*lt_rb_w_ne; // right bottom

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ void ldu_micro_bilinear_multi_output(const scalar_t *bottom_data, 
                                        const int data_width,
                                        const int height, 
                                        const int width, 
                                        scalar_t h, 
                                        scalar_t w,
                                        scalar_t *output_ptr,
                                        scalar_t *output_lt_ptr,
                                        scalar_t *output_lb_ptr,
                                        scalar_t *output_rt_ptr,
                                        scalar_t *output_rb_ptr)
{
  scalar_t upper_bound_x = height - 1.;
  scalar_t upper_bound_y = width - 1.;

  if (h > upper_bound_x)
      h = upper_bound_x;
  if (h < 0.) 
      h = 0.;
  if (w > upper_bound_y) 
      w = upper_bound_y;
  if (w < 0.) 
      w = 0.;
    
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = ceil(h);
  int w_high = ceil(w);
  
  scalar_t lt_rb_h_ne = 1.;
  scalar_t lt_rb_w_ne = 1.;
  
  if (h_low == h_high)
      lt_rb_h_ne = 0.;
  if (w_low == w_high)
      lt_rb_w_ne = 0.;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1. - lh, hw = 1. - lw;
  
  scalar_t v1 = bottom_data[h_low * data_width + w_low]; // left top
  scalar_t v2 = bottom_data[h_low * data_width + w_high]; // right top
  scalar_t v3 = bottom_data[h_high * data_width + w_low]; // left bottom
  scalar_t v4 = bottom_data[h_high * data_width + w_high]; // right bottom

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 * lt_rb_w_ne + w3 * v3 * lt_rb_h_ne + w4 * v4 * lt_rb_h_ne *lt_rb_w_ne);
  
  *output_ptr = val;
  *output_lt_ptr = v1;
  *output_lb_ptr = v3;
  *output_rt_ptr = v2;
  *output_rb_ptr = v4;
}

template <typename scalar_t>
__device__ void lau_micro_coord_bilinear(const scalar_t grad_out,
                                        const scalar_t *bottom_data, 
                                        const int data_width,
                                        const int height, 
                                        const int width, 
                                        scalar_t h, 
                                        scalar_t w,
                                        scalar_t *grad_x_ptr,
                                        scalar_t *grad_y_ptr,
                                        scalar_t *grad_input_ptr)
{
  scalar_t upper_bound_x = height - 1;
  scalar_t upper_bound_y = width - 1;
  
  const scalar_t H = h;
  const scalar_t W = w;

  if (h > upper_bound_x) 
      h = upper_bound_x;
  if (h < 0.) 
      h = 0.;
  if (w > upper_bound_y) 
      w = upper_bound_y;
  if (w < 0.) 
      w = 0.;
    
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = ceil(h);
  int w_high = ceil(w);
  
  scalar_t lt_rb_h_ne = 1.;
  scalar_t lt_rb_w_ne = 1.;
  
  if (h_low == h_high)
      lt_rb_h_ne = 0.;
  if (w_low == w_high)
      lt_rb_w_ne = 0.;

  scalar_t lh = h - h_low; // x - lower_x
  scalar_t lw = w - w_low; // y - lower_y
  scalar_t hh = 1. - lh;   // upper_x - x
  scalar_t hw = 1. - lw;   // upper_y - y
  
  const int lt_coords = h_low * data_width + w_low;
  const int rt_coords = h_low * data_width + w_high;
  const int lb_coords = h_high * data_width + w_low;
  const int rb_coords = h_high * data_width + w_high;

  scalar_t v1 = bottom_data[lt_coords]; // left top
  scalar_t v2 = bottom_data[rt_coords]*lt_rb_w_ne; // right top
  scalar_t v3 = bottom_data[lb_coords]*lt_rb_h_ne; // left bottom
  scalar_t v4 = bottom_data[rb_coords]*lt_rb_h_ne*lt_rb_w_ne; // right bottom
  
  // input gradients
  atomicAdd(grad_input_ptr + lt_coords, grad_out * hh * hw);
  if (lt_rb_w_ne == 1.)
    atomicAdd(grad_input_ptr + rt_coords, grad_out * hh * lw);
  
  // coord gradients
  *grad_x_ptr = 0.;
  *grad_y_ptr = 0.;
  
  if (H <= upper_bound_x && H >= 0.)
      *grad_x_ptr += (-1. * hw * v1 - 1. * lw * v2 + hw * v3 + lw * v4) * grad_out;     
  if (W <= upper_bound_y && W >= 0.)
      *grad_y_ptr += (-1. * hh * v1 + hh * v2 - 1. * lh * v3 + lh * v4) * grad_out; 
  
  // input gradients
  if (lt_rb_h_ne == 1.)
    atomicAdd(grad_input_ptr + lb_coords, grad_out * lh * hw);
  if (lt_rb_h_ne == 1. && lt_rb_w_ne == 1.)
    atomicAdd(grad_input_ptr + rb_coords, grad_out * lh * lw);  
}

template <typename scalar_t>
__global__ void lau_bilinear_gpu_kernel(const int n, 
                                            const scalar_t *data_im, 
                                            const scalar_t *data_offset_x, 
                                            const scalar_t *data_offset_y, 
                                            const int height, 
                                            const int width, 
                                            const int batch_size, 
                                            const int num_channels, 
                                            const int height_out, 
                                            const int width_out, 
                                            scalar_t *data_out)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // NOTE(Jiarui XU): different from CharlesShang's implementation, col_buffer is of shape (N, c*kw*kh, oh * ow)
    // here columns is of shape (c*kw*kh, N, oh, ow), need to adapt axis

    // index index of output matrix
    const int size_col = width_out * height_out;
    const int c_col = index / size_col;
    const int h_col = (index - c_col*size_col) / width_out;
    const int w_col = index % width_out;
    
    const int c_im = c_col;
    const int offset_ = (c_col * height_out + h_col) * width_out + w_col;
    
    scalar_t *data_col_ptr = data_out + offset_;

    const scalar_t *data_im_ptr = data_im + c_im * height * width;
    
    const scalar_t offset_h = data_offset_x[offset_];
    const scalar_t offset_w = data_offset_y[offset_];
    scalar_t val = static_cast<scalar_t>(0);
    const scalar_t k_h = height_out / height;
    const scalar_t k_w = width_out / width;
    const scalar_t h_im = h_col / k_h + offset_h;
    const scalar_t w_im = w_col / k_w + offset_w;

    val = lau_micro_bilinear(data_im_ptr, width, height, width, h_im, w_im);
    
    *data_col_ptr = val;
  }
}

template <typename scalar_t>
__global__ void ldu_bilinear_gpu_kernel(const int n, 
                                            const scalar_t *data_im, 
                                            const int height, 
                                            const int width, 
                                            const int batch_size, 
                                            const int num_channels, 
                                            const int height_out, 
                                            const int width_out, 
                                            scalar_t *data_out)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // NOTE(Jiarui XU): different from CharlesShang's implementation, col_buffer is of shape (N, c*kw*kh, oh * ow)
    // here columns is of shape (c*kw*kh, N, oh, ow), need to adapt axis

    // index index of output matrix
    const int size_col = width_out * height_out;
    const int c_col = index / size_col;
    const int h_col = (index - c_col*size_col) / width_out;
    const int w_col = index % width_out;
    
    const int c_im = c_col;
    const int offset_ = (c_col * height_out + h_col) * width_out + w_col;
    
    scalar_t *data_col_ptr = data_out + offset_;

    const scalar_t *data_im_ptr = data_im + c_im * height * width;
    
    scalar_t val = static_cast<scalar_t>(0);
    const scalar_t k_h = height_out / height;
    const scalar_t k_w = width_out / width;
    const scalar_t h_im = h_col / k_h;
    const scalar_t w_im = w_col / k_w;

    val = lau_micro_bilinear(data_im_ptr, width, height, width, h_im, w_im);
    
    *data_col_ptr = val;
  }
}

template <typename scalar_t>
__global__ void ldu_bilinear_multi_output_gpu_kernel(const int n, 
                                            const scalar_t *data_im, 
                                            const int height, 
                                            const int width, 
                                            const int batch_size, 
                                            const int num_channels, 
                                            const int height_out, 
                                            const int width_out, 
                                            scalar_t *data_out,
                                            scalar_t *data_out_lt,
                                            scalar_t *data_out_lb,
                                            scalar_t *data_out_rt,
                                            scalar_t *data_out_rb)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // NOTE(Jiarui XU): different from CharlesShang's implementation, col_buffer is of shape (N, c*kw*kh, oh * ow)
    // here columns is of shape (c*kw*kh, N, oh, ow), need to adapt axis

    // index index of output matrix
    const int size_col = width_out * height_out;
    const int c_col = index / size_col;
    const int h_col = (index - c_col*size_col) / width_out;
    const int w_col = index % width_out;
    
    const int c_im = c_col;
    const int offset_ = (c_col * height_out + h_col) * width_out + w_col;
    
    scalar_t *data_col_ptr = data_out + offset_;
    scalar_t *data_col_lt_ptr = data_out_lt + offset_;
    scalar_t *data_col_lb_ptr = data_out_lb + offset_;
    scalar_t *data_col_rt_ptr = data_out_rt + offset_;
    scalar_t *data_col_rb_ptr = data_out_rb + offset_;

    const scalar_t *data_im_ptr = data_im + c_im * height * width;
    
    const scalar_t k_h = height_out / height;
    const scalar_t k_w = width_out / width;
    const scalar_t h_im = h_col / k_h;
    const scalar_t w_im = w_col / k_w;

    ldu_micro_bilinear_multi_output(data_im_ptr, width, height, width, h_im, w_im,
        data_col_ptr, data_col_lt_ptr, data_col_lb_ptr, data_col_rt_ptr, data_col_rb_ptr);
  }
}

template <typename scalar_t>
__global__ void lau_bilinear_coord_gpu_kernel(const int n,
                                                             const scalar_t *grad_col, 
                                                             const scalar_t *data_im,
                                                             const scalar_t *data_offset_x,
                                                             const scalar_t *data_offset_y,
                                                             const int channels, 
                                                             const int height,
                                                             const int width,
                                                             const int batch_size,
                                                             const int height_out, 
                                                             const int width_out,
                                                             scalar_t *grad_offset_x,
                                                             scalar_t *grad_offset_y,
                                                             scalar_t *grad_input)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int size_col = width_out * height_out;
    const int c_col = index / size_col;
    const int h_col = (index - c_col*size_col) / width_out;
    const int w_col = index % width_out;
    
    const int c_im = c_col;
    
    const int offset_ = (c_col * height_out + h_col) * width_out + w_col;

    scalar_t *grad_offset_x_ptr = grad_offset_x + offset_;
    scalar_t *grad_offset_y_ptr = grad_offset_y + offset_;
    const scalar_t *grad_out_ptr = grad_col + offset_;

    const scalar_t *data_im_ptr = data_im + c_im * height * width;
    scalar_t *grad_im_ptr = grad_input + c_im * height * width;
    
    const scalar_t offset_h = data_offset_x[offset_];
    const scalar_t offset_w = data_offset_y[offset_];
    
    const scalar_t k_h = height_out / height;
    const scalar_t k_w = width_out / width;
    const scalar_t h_im = h_col / k_h + offset_h;
    const scalar_t w_im = w_col / k_w + offset_w;
    
    const scalar_t grad_out = *grad_out_ptr;

    lau_micro_coord_bilinear(grad_out, data_im_ptr, width, height, width, h_im, w_im, grad_offset_x_ptr, grad_offset_y_ptr, grad_im_ptr);
  }
}

template <typename scalar_t>
void lau_bilinear_cuda(cudaStream_t stream,
  const scalar_t* data_im, const scalar_t* data_offset_x, const scalar_t* data_offset_y,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_out, const int width_out, scalar_t* data_out) {
  // num_axes should be smaller than block size
  const int num_kernels = channels * batch_size * height_out * width_out;
  lau_bilinear_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(num_kernels, data_im, data_offset_x, data_offset_y, height_im, 
                            width_im, batch_size, channels, height_out, width_out, data_out);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void ldu_bilinear_cuda(cudaStream_t stream,
  const scalar_t* data_im, const int batch_size, const int channels, const int height_im, const int width_im, const int height_out, const int width_out, scalar_t* data_out) {
  // num_axes should be smaller than block size
  const int num_kernels = channels * batch_size * height_out * width_out;
  ldu_bilinear_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(num_kernels, data_im, height_im, 
                            width_im, batch_size, channels, height_out, width_out, data_out);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void ldu_bilinear_multi_output_cuda(cudaStream_t stream,
  const scalar_t* data_im, const int batch_size, const int channels, const int height_im, const int width_im, const int height_out, const int width_out, scalar_t* data_out, scalar_t* data_out_lt, scalar_t* data_out_lb, scalar_t* data_out_rt, scalar_t* data_out_rb) {
  // num_axes should be smaller than block size
  const int num_kernels = channels * batch_size * height_out * width_out;
  ldu_bilinear_multi_output_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(num_kernels, data_im, height_im, 
                            width_im, batch_size, channels, height_out, width_out, 
                                data_out, data_out_lt, data_out_lb, data_out_rt, data_out_rb);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void lau_bilinear_cuda_backward(cudaStream_t stream,
  const scalar_t* grad_col, const scalar_t* data_im, const scalar_t* data_offset_x, 
  const scalar_t* data_offset_y, const int batch_size, const int channels, 
  const int height_im, const int width_im, const int height_col, const int width_col, 
  scalar_t* grad_offset_x, scalar_t* grad_offset_y, scalar_t* grad_input) {
  const int num_kernels = batch_size * height_col * width_col * channels;
  lau_bilinear_coord_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, grad_col, data_im, data_offset_x, data_offset_y, 
        channels, height_im, width_im, batch_size, height_col, width_col, 
        grad_offset_x, grad_offset_y, grad_input);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in location_aware_upsampling_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}
