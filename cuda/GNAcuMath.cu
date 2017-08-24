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
  if (!(x >= n || x < 0 || y >= n || y < 0))  mat[x + n * y] = (x == y) ? 1.0 : 0.0;  
}

void cuInverseMat(int matSize, double* InMat, double* OutMat) {
  cublasHandle_t handle;
  cublasStatus_t ret;
  cudaError_t err;
  ret = cublasCreate(&handle);
  if(ret!=CUBLAS_STATUS_SUCCESS){
    printf("error code %d, line(%d)\n", ret, __LINE__);
    exit(EXIT_FAILURE);
  }

  double* devInMat;
  double* devOutMat;
  cudaMalloc((void**)&devInMat,  matSize*matSize*sizeof(double));
  cudaMalloc((void**)&devOutMat,  matSize*matSize*sizeof(double));

  GenIdentity<<<1, dim3(matSize, matSize)>>>(matSize, devOutMat);
  
  err = cudaMemcpy(devInMat, InMat, matSize*matSize*sizeof(double), cudaMemcpyHostToDevice);
  if(err!=cudaSuccess) {
#ifdef DEBUG
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
#endif
    exit(EXIT_FAILURE);
  }

  double alpha = 1.0;
  /**
  *  Solve A*x = alpha * B to invert matrix. In this case B is Identity, alpha == 1.
  */
  ret = cublasDtrsm_v2(handle,
          CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
          matSize, matSize, &alpha, devInMat, matSize, devOutMat, matSize);

  if(ret!=CUBLAS_STATUS_SUCCESS) {
#ifdef  DEBUG
    printf("error code %d, line(%d)\n", ret, __LINE__);
#endif
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(OutMat, devOutMat, matSize*matSize*sizeof(double), cudaMemcpyDeviceToHost);

  if(err!=cudaSuccess) {
#ifdef DEBUG
    printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
#endif
    exit(EXIT_FAILURE);
  }

  cudaFree(devInMat);
  cudaFree(devOutMat);
}

/*
int main() {
  const int matSize = 4096;
  double *InMat = new double[matSize*matSize];
  double *OutMat = new double[matSize*matSize];
  for(int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      if (i == j){ OutMat [i+matSize*j] = 1.0;  InMat [i+matSize*j] = 2.0;}
      else {OutMat [i+matSize*j] = 0.0;  InMat [i+matSize*j] = 0.0;}
//      if (!(j >= 3/2 && i < 3/2)) { 
//         InMat[i+3*j] = 2.0;
//      } 
//      else { InMat[i+3*j] = 0.0; }
    }
  }
   printf("OutMat:\n");
  //int matSize = 3;
//  for (int i = 0; i < matSize; i++) {
//    for (int j = 0; j < matSize; j++) {
//      printf("%lf ", OutMat[i + matSize*j]);
//    }
//    printf("\n");
//  }

  cuInverseMat(matSize, InMat, OutMat);
  return 0;
}
*/
