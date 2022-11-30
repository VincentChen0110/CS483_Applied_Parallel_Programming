#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define MASK_WIDTH 3
#define MASK_RADIUS 1
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int Row = by*TILE_WIDTH+ty;
  int Col = bx*TILE_WIDTH+tx;
  int Hei = bz*TILE_WIDTH+tz;

  int Row_s = Row - MASK_RADIUS;
  int Col_s = Col - MASK_RADIUS;
  int Hei_s = Hei - MASK_RADIUS;
  
  __shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

  if ((Row_s >=0 && Row_s < y_size) && (Col_s>=0 && Col_s < x_size) && (Hei_s >=0 && Hei_s < z_size)){
    N_ds[tz][ty][tx] = input[Hei_s*y_size*x_size + Row_s*x_size + Col_s];
  }
  else{
   N_ds[tz][ty][tx] = 0.0; 
  }
  
  __syncthreads();
  
  float Pvalue = 0;
  if(tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH){
    for(int i = 0; i < MASK_WIDTH; i++) { 
      for(int j = 0; j < MASK_WIDTH; j++) {
        for (int k = 0; k < MASK_WIDTH; k++ ) {
          Pvalue += deviceKernel[i][j][k] * N_ds[i+tz][j+ty][k+tx];  
        }
      }
    }
    __syncthreads();
    
    if( Row < y_size && Col < x_size && Hei < z_size){
      output[Hei*y_size*x_size + Row*x_size + Col] = Pvalue;
    }
  }





}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInput, (inputLength-3) *sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength-3) *sizeof(float));
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  cudaMemcpy(deviceInput, hostInput+3, (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float), 0, cudaMemcpyHostToDevice);
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((1.0*x_size)/TILE_WIDTH), ceil((1.0*y_size)/TILE_WIDTH),ceil((1.0*z_size)/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH+MASK_WIDTH-1, TILE_WIDTH+MASK_WIDTH-1, TILE_WIDTH+MASK_WIDTH-1);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size,
                       y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
