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
    //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int WGrid = ceil(1.0*Width/ TILE_WIDTH);
    int h = (blockIdx.y/WGrid)*TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y%WGrid)*TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;
    int m = blockIdx.x;

    float result = 0;
    for(int c = 0; c < Channel; c++){ // sum over input feature map
        for(int p  = 0; p < K; p++){ // sum over kxk filter
            for(int q = 0; q < K; q++){ 
            //output[b][m][h][w] += input[b][c][h + p][w + q] * k[m][c][p][q]
            result += in_4d(b, c, h+p, w+q)*mask_4d(m, c, p, q);
            }
        }
    }

    if( h < Height_out && w < Width_out){
      out_4d(b, m, h, w) = result;
    }
    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
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

    const int stream_num = 10;
    const int Height_Out = Height - K+1;
    const int Width_Out = Width - K+1;

    float* host_out_tmp = (float*) host_output;
    
    int Out_Size = Batch*Map_out*Height_Out*Width_Out/stream_num;
    int In_Size = Batch*Channel*Height*Width/stream_num;
    int Kernel_Size = Map_out*Channel*K*K;

    int W_grid = (Width_Out+TILE_WIDTH-1)/TILE_WIDTH;
    int H_grid = (Height_Out+TILE_WIDTH -1)/TILE_WIDTH;
    int Grid = W_grid*H_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, Grid, Batch/stream_num);

    // allocate
    cudaMalloc((void **) device_output_ptr, Batch*Map_out*Height_Out*Width_Out*sizeof(float));
    cudaMalloc((void **) device_input_ptr, Batch*Channel*Height*Width*sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Kernel_Size*sizeof(float));

    cudaStream_t stream[stream_num];
    for(int i =0;i<stream_num; i++){
        cudaStreamCreate(&stream[i]);
    }
    cudaMemcpyAsync(*device_mask_ptr, host_mask, Kernel_Size*sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    for(int i = 0; i<stream_num; i++){
        int in_offset = In_Size*i;
        int out_offset = Out_Size*i;
        cudaMemcpyAsync((*device_input_ptr)+in_offset, host_input + in_offset, In_Size*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        conv_forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>((*device_output_ptr)+out_offset, (*device_input_ptr)+in_offset, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        cudaMemcpyAsync(host_out_tmp+out_offset, (*device_output_ptr)+out_offset, Out_Size*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for(int i = 0; i< stream_num; i++){
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(device_output_ptr);
    cudaFree(device_input_ptr);
    cudaFree(device_mask_ptr);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel

    // int Grid = (ceil(1.0*Width/ TILE_WIDTH)) * (ceil(1.0*Height/ TILE_WIDTH));
    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(Map_out, Grid, Batch);

    // conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    return ;

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // const int Height_Out = Height - K+1;
    // const int Width_Out = Width - K+1;

    // int Out_Size = Batch*Map_out*Height_Out*Width_Out;
    // // Copy the output back to host
    // cudaMemcpy(host_output, device_output, Out_Size * sizeof(float), cudaMemcpyDeviceToHost);
    // // Free device memory
    // cudaFree(device_output);
    // cudaFree(device_input);
    // cudaFree(device_mask);
    return ;

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