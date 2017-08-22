#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <typeinfo>
#include <iostream>
#include <cuda.h>


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

  err = cudaMemcpy(devInMat, InMat, matSize*matSize*sizeof(double), cudaMemcpyHostToDevice);
  if(err!=cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
  }

  err = cudaMemcpy(devOutMat, OutMat, matSize*matSize*sizeof(double), cudaMemcpyHostToDevice);
  if(err!=cudaSuccess) {
    printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
  }

  double alpha = 1.0;
  ret = cublasDtrsm_v2(handle,
          CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
          matSize, matSize, &alpha, devInMat, matSize, devOutMat, matSize);

  if(ret!=CUBLAS_STATUS_SUCCESS) {
    printf("error code %d, line(%d)\n", ret, __LINE__);
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(OutMat, devOutMat, matSize*matSize*sizeof(double), cudaMemcpyDeviceToHost);

  if(err!=cudaSuccess) {
    printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
  }

  printf("InMat:\n");
/*
  for (int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      printf("%lf ", InMat[i + matSize*j]);
      
    }
    printf("\n");
  }
  printf("OutMat:\n");
  for (int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      printf("%lf ", OutMat[i + matSize*j]);
    }
    printf("\n");
  }
*/
  cudaFree(devInMat);
  cudaFree(devOutMat);

}

int main() {
  const int matSize = 5000;
  double *InMat = new double[matSize*matSize];
  double *OutMat = new double[matSize*matSize];
  for(int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      if (i == j){ OutMat [i+matSize*j] = 1.0;  InMat [i+matSize*j] = 2.0;}
      else {OutMat [i+matSize*j] = 0.0;  InMat [i+matSize*j] = 0.0;}
/*      if (!(j >= 3/2 && i < 3/2)) { 
         InMat[i+3*j] = 2.0;
      } 
      else { InMat[i+3*j] = 0.0; }
*/    }
  }
   printf("OutMat:\n");
  //int matSize = 3;
  /*for (int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      printf("%lf ", OutMat[i + matSize*j]);
    }
    printf("\n");
  }
*/
  cuInverseMat(matSize, InMat, OutMat);
  return 0;
}

/*int cuInverseMatInPlace(int matSize, double* InvMat) {

}*/
