// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this
#define SECTION_SIZE BLOCK_SIZE * 2

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *sum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[SECTION_SIZE];
  
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int i = 2*blockIdx.x*blockDim.x + threadIdx.x;

  if(i < len){
    T[tx] = input[i];
  }
  else{
    T[tx] = 0;
  }

  __syncthreads();

  if(i+blockDim.x < len){
    T[tx+blockDim.x] = input[i+blockDim.x];
  }
  else{
    T[tx+blockDim.x] = 0;
  }

  __syncthreads();

  // FirstScan
  int stride = 1;
  while(stride < SECTION_SIZE){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if(index<SECTION_SIZE && (index-stride)>=0){
      T[index] += T[index-stride];
    }
    stride *= 2;
  }

  // PostScan
  stride = BLOCK_SIZE/2;
  while(stride > 0){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if((index+stride) < SECTION_SIZE){
      T[index+stride] += T[index];
    }
    stride /= 2;

  }

  __syncthreads();

  //Write to output
  if(i<len){
    output[i] = T[tx];
  }
  if(i+blockDim.x < len){
    output[i+blockDim.x] = T[tx+blockDim.x];
  }

  //Store sum
  if(sum){
    if(tx==0){
      sum[bx] = T[SECTION_SIZE-1];
    }
  }

}

__global__ void add(float *input, float *output, int len){
  
  // for( int i = 0 ;i< len; i++ ) {
  //     printf("input:%f output:%f", input[i] , output[i]);
  //  }
   
  __shared__ float tmp_sum;
  if (threadIdx.x == 0){
    if (blockIdx.x == 0)
      tmp_sum = 0;
    else
      tmp_sum = input[blockIdx.x - 1];
  }
  __syncthreads();
  
  for(int k = 0; k < 2; ++k){
    int tile = (blockIdx.x * blockDim.x * 2) + threadIdx.x + (k * BLOCK_SIZE);
    if(tile < len){
      output[tile] += tmp_sum;
    }
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *auxSum;   //store sum
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //allocate aux
  int numBlocks = ceil((numElements*1.0)/(SECTION_SIZE));
  wbCheck(cudaMalloc((void **)&auxSum, numBlocks * sizeof(float)));
  
  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numElements/(BLOCK_SIZE*1.0)),1,1);  
  dim3 DimBlock(BLOCK_SIZE,1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements, auxSum);
  cudaDeviceSynchronize();

  dim3 DimGrid1(1, 1, 1);
  scan<<<DimGrid1, DimBlock>>>(auxSum, auxSum, numBlocks, NULL);
  cudaDeviceSynchronize();

  add<<<DimGrid, DimBlock>>>(auxSum, deviceOutput, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
