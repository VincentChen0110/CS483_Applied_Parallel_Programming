#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 32

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int input_tile_width = TILE_WIDTH + K - 1;
    //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[input_tile_width*input_tile_width];
    const int W_x = blockDim.x + K - 1;
    const int H_x = blockDim.y + K - 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_2d(i1, i0) W_shared[(i1) * K + (i0)]
    #define in_2d(i1, i0) X_shared[(i1) * W_x + (i0)]

    // Insert your GPU convolution kernel code here
    
    // int WGrid = ceil(1.0*Width/ TILE_WIDTH);
    // int h = (blockIdx.y/WGrid)*TILE_WIDTH + threadIdx.y;
    // int w = (blockIdx.y%WGrid)*TILE_WIDTH + threadIdx.x;
    // int b = blockIdx.z;
    // int m = blockIdx.x;
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int W_grid = ceil(1.0*Width/ TILE_WIDTH);
    const int h_base = blockDim.y * (blockIdx.z / W_grid);
    const int w_base = blockDim.x * (blockIdx.z % W_grid);
    const int h_thread = threadIdx.y;
    const int w_thread = threadIdx.x;
    const int h = h_base + h_thread;
    const int w = w_base + w_thread;
    float res = 0.0;

    for(int c = 0; c < Channel; ++c) {
      // Load slice of kernel for m, c
      __syncthreads();
      if (h_thread < K && w_thread < K)
        mask_2d(h_thread, w_thread) = mask_4d(m, c, h_thread, w_thread);

      // Load slice of x for b, c
      for (int i = h_thread; i < H_x; i += blockDim.y)
        for (int j = w_thread; j < W_x; j += blockDim.x)
          in_2d(i, j) = in_4d(b, c, h_base + i, w_base + j);

      __syncthreads();
      for (int p = 0; p < K; p++) { 
        for (int q = 0; q < K; q++) {
          res += in_2d(h_thread + p, w_thread + q) * mask_2d(p, q);
        }
      }
    }

    if (h < Height_out && w < Width_out)
      out_4d(b, m, h, w) = res;



    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef mask_2d
    #undef in_2d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_Out = Height - K+1;
    const int Width_Out = Width - K+1;

    int Out_Size = Batch*Map_out*Height_Out*Width_Out;
    int In_Size = Batch*Channel*Height*Width;
    int Kernel_Size = Map_out*Channel*K*K;
    // allocate
    cudaMalloc((void **) device_output_ptr, Out_Size*sizeof(float));
    cudaMalloc((void **) device_input_ptr, In_Size*sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Kernel_Size*sizeof(float));
    // memcpy
    cudaMemcpy(*device_input_ptr, host_input, In_Size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Kernel_Size*sizeof(float), cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel

    int Z_grid = ceil(1.0*Width/ TILE_WIDTH)*ceil(1.0*Width/ TILE_WIDTH);
    int elems_x_shared = (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1);
    int elems_k_shared = K*K;
    const size_t sharemem = sizeof(float) * (elems_x_shared + elems_k_shared);

    // Set the kernel dimensions and call the kernel
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH);
    dim3 gridDim(Batch, Map_out, Z_grid);

    conv_forward_kernel<<<gridDim, blockDim, sharemem>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_Out = Height - K+1;
    const int Width_Out = Width - K+1;

    int Out_Size = Batch*Map_out*Height_Out*Width_Out;
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, Out_Size * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
