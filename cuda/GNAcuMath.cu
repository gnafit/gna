#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <typeinfo>
#include <iostream>
#include <cuda.h>
#include "GNAcuMath.h"

/**
  *  Generation of Identity matrix on GPU memory
  *  TODO: find an optimal grid and block sizes!
  */
__global__ void GenIdentity (int n, double * mat) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if (x < n && y < n)
  mat[x + n * y] = (x == y) ? 1.0 : 0.0; 
}

void cuInverseMat(int matSize, double* InMat, double* OutMat) {
  cudaSetDevice(0);
  cublasHandle_t handle;
  cublasStatus_t ret;
  cudaError_t err;
  
  cudaStream_t stream1, stream2;
  cudaStreamCreate ( &stream1);
  cudaStreamCreate ( &stream2);

  ret = cublasCreate(&handle);
  if(ret!=CUBLAS_STATUS_SUCCESS){
    exit(EXIT_FAILURE);
  }

  double* devInMat;
  double* devOutMat;
  cudaMalloc((void**)&devInMat,  matSize*matSize*sizeof(double));
  cudaMalloc((void**)&devOutMat,  matSize*matSize*sizeof(double));

  GenIdentity<<<dim3(matSize/32 + 1, matSize/32 + 1), dim3(32,32), 0, stream2>>>(matSize, devOutMat);
  //cudaDeviceSynchronize();

  err = cudaMemcpyAsync(devInMat, InMat, matSize*matSize*sizeof(double), cudaMemcpyHostToDevice, stream1);
  if(err!=cudaSuccess) {
    exit(EXIT_FAILURE);
  }

  double alpha = 1.0;
/**
  *  Solve A*x = alpha * B to invert matrix. In this case B is Identity, alpha == 1.
  */
  ret = cublasDtrsm_v2(handle,
          CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
          matSize, matSize, &alpha, devInMat, matSize, devOutMat, matSize);
  cudaDeviceSynchronize();
  if(ret!=CUBLAS_STATUS_SUCCESS) {
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpyAsync(OutMat, devOutMat, matSize*matSize*sizeof(double), cudaMemcpyDeviceToHost, stream1);

  if(err!=cudaSuccess) {
    exit(EXIT_FAILURE);
  }

  cudaFree(devInMat);
  cudaFree(devOutMat);
}

/*int main () {
  int n = 200;
  double* inM = new double[n*n];
  double* outM = new double[n*n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        inM[i + n*j] = i + j + 1;
        outM[i + n*j] = 1;
      }
      else {
        inM[i + n*j] = 0;
        outM[i + n*j] = 0;
      }
    }
  }

  for(int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (outM[i + j*n] != 0)
      std::cout << inM[i + j*n] << " ";
    }
  //  std::cout << std::endl;
  }

  cuInverseMat(n, inM,  outM);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (outM[i + j*n] != 0) 
      std::cout << outM[i + j*n] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
*/
