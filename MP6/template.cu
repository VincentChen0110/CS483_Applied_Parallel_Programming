// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void float2char(float* inputImage, unsigned char *outputImage, int size){
  int Row = blockDim.x*blockIdx.x + threadIdx.x;
  if(Row < size){
    outputImage[Row] = (unsigned char) (255*inputImage[Row]);
  }
}

__global__ void rgb2gray(unsigned char* inputImage, unsigned char* outputImage, int size){
  int Row = blockDim.x*blockIdx.x + threadIdx.x;
  if(Row< size){
    unsigned char r = inputImage[3*Row];
    unsigned char g = inputImage[3*Row+1];
    unsigned char b = inputImage[3*Row+2];
    outputImage[Row] = (unsigned char) (0.21*r+0.71*g+0.07*b);
  }
}

__global__ void computeHist(unsigned char *buffer, unsigned int *histo, int size){
  
  __shared__ unsigned int H[HISTOGRAM_LENGTH];

  if (threadIdx.x < HISTOGRAM_LENGTH){
    H[threadIdx.x] = 0;
  }
  __syncthreads();
  int Row = blockDim.x*blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (Row<size){
    atomicAdd( &(H[buffer[Row]]), 1);
    Row += stride;
  }
  __syncthreads();
  if (threadIdx.x < HISTOGRAM_LENGTH){
    atomicAdd(&(histo[threadIdx.x]), H[threadIdx.x]);
  }
}

__global__ void histCdf(unsigned int *histo, float *cdf, int size){
  
  __shared__ float T[2*HISTOGRAM_LENGTH];
  
  int tx = threadIdx.x;
  int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
  if(i < HISTOGRAM_LENGTH){
    T[tx] = histo[i];
  }
  else{
    T[tx] = 0.0;
  }

  __syncthreads();

  if(i+blockDim.x < HISTOGRAM_LENGTH){
    T[tx+blockDim.x] = histo[i+blockDim.x];
  }
  else{
    T[tx+blockDim.x] = 0.0;
  }

  __syncthreads();

  // FirstScan
  int stride = 1;
  while(stride <= HISTOGRAM_LENGTH){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if(index<HISTOGRAM_LENGTH && (index-stride)>=0){
      T[index] += T[index-stride];
    }
    stride *= 2;
  }

  // PostScan
  stride = HISTOGRAM_LENGTH/2;
  while(stride > 0){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if((index+stride) < HISTOGRAM_LENGTH){
      T[index+stride] += T[index];
    }
    stride /= 2;

  }

  __syncthreads();
  //Write to output
  if(i<HISTOGRAM_LENGTH){
    cdf[i] = T[tx]/((float) size);
  }
  if(i+blockDim.x < HISTOGRAM_LENGTH){
    cdf[i+blockDim.x] = T[tx+blockDim.x]/((float)size);
  }

}

__global__ void histEqual(unsigned char *imageIn, float *imageOut, float *cdf, int size){
  int Row = blockDim.x*blockIdx.x + threadIdx.x;
  if (Row < size){
    float tmp = 255*(cdf[imageIn[Row]] - cdf[0])/(1-cdf[0])/255.0;
    imageOut[Row] = (float) min(max(tmp, 0.0), 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceIn;
  unsigned char *deviceChar;
  unsigned char *deviceGray;
  unsigned int *deviceHisto;
  float *deviceCdf;
  float *deviceOut;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  // allocate mem
  int rgbSize = imageWidth * imageHeight * imageChannels;
  int gSize = imageWidth * imageHeight * 1;
  cudaMalloc((void **) &deviceIn, rgbSize*sizeof(float));
  cudaMalloc((void **) &deviceChar, rgbSize*sizeof(unsigned char));
  cudaMalloc((void **) &deviceGray, gSize*sizeof(unsigned char));
  cudaMalloc((void **) &deviceHisto, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMalloc((void **) &deviceCdf, HISTOGRAM_LENGTH*sizeof(float));
  cudaMalloc((void **) &deviceOut, rgbSize*sizeof(float));
  // copy mem
  cudaMemset((void *) deviceHisto, 0, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMemset((void *) deviceCdf, 0, HISTOGRAM_LENGTH*sizeof(float));
  cudaMemcpy(deviceIn, hostInputImageData, rgbSize*sizeof(float), cudaMemcpyHostToDevice);
  //define block
  dim3 dimBlock(HISTOGRAM_LENGTH);
  dim3 dimGrid((rgbSize-1)/HISTOGRAM_LENGTH + 1);
  
  //lunch kernel
  
  float2char<<<dimGrid, dimBlock>>>(deviceIn, deviceChar, rgbSize);
  rgb2gray<<<dimGrid, dimBlock>>>(deviceChar, deviceGray, gSize);
  computeHist<<<dimGrid, dimBlock>>>(deviceGray, deviceHisto, gSize);
  histCdf<<<dimGrid, dimBlock>>>(deviceHisto, deviceCdf, gSize);
  histEqual<<<dimGrid, dimBlock>>>(deviceChar, deviceOut, deviceCdf, rgbSize);

  cudaMemcpy(hostOutputImageData, deviceOut, rgbSize*sizeof(float), cudaMemcpyDeviceToHost);

  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceIn);
  cudaFree(deviceChar);
  cudaFree(deviceGray);
  cudaFree(deviceHisto);
  cudaFree(deviceCdf);
  cudaFree(deviceOut);
  return 0;
}
