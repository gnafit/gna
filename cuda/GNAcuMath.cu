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


/**
* cuBLAS multiplier wrapper for GNA 
*/
void cuMultiplyMat(int m, int n, int k, double* InA, double* InB, double* OutC) {
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
  double* devA; double* devB; double* devC;
  cudaMalloc((void**)&devA, m * k * sizeof(double));
  cudaMalloc((void**)&devB, k * n * sizeof(double));
  cudaMalloc((void**)&devC, m * n * sizeof(double));
  
  cudaMemcpyAsync(devA, InA, m * k * sizeof(double), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(devB, InB, k * n * sizeof(double), cudaMemcpyHostToDevice, stream2);
  cudaMemset(devC, 0, m * n * sizeof(double));
  double alpha = 1, beta = 0;
  cudaDeviceSynchronize();  
  cublasDgemm(handle, 
              CUBLAS_OP_N,  CUBLAS_OP_N,
              m, n, k,
              &alpha,
              devA, m,
              devB, k,
              &beta,
              devC, m );
  
  err = cudaMemcpyAsync(OutC, devC, m * n * sizeof(double), cudaMemcpyDeviceToHost, stream1);

  if(err!=cudaSuccess) {
    exit(EXIT_FAILURE);
  }
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cublasDestroy(handle);
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
}


/**
* cuBLAS linear system solver wrapper for GNA. A is lower triangular.
*/
void cuSolveLowerLS(int m, int n, double* A, double* B) {
//printf("BEF fffff \n");
//  cudaSetDevice(0);
//printf("AF set dev \n");

  cublasHandle_t handle;
  cublasStatus_t ret;
  cudaError_t err;
//printf("BEF str\n");
  cudaStream_t stream1, stream2;
  cudaStreamCreate ( &stream1);
  cudaStreamCreate ( &stream2);

  ret = cublasCreate(&handle);
  if(ret!=CUBLAS_STATUS_SUCCESS){
    printf("cublasCreate(&handle)");
    exit(EXIT_FAILURE);
  }
//printf("BEF cumalloc \n");
  double* devA;
  double* devB;
  cudaMalloc((void**)&devA,  m*m*sizeof(double));
  cudaMalloc((void**)&devB,  m*n*sizeof(double));
//printf("AF cumalloc \n");

  cudaMemcpyAsync(devA, A, m*m*sizeof(double), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(devB, B, m*n*sizeof(double), cudaMemcpyHostToDevice, stream2);
  
  double alpha = 1.0;
/**
  *  Solve A*x =  B.
  */
  ret = cublasDtrsm_v2(handle,
          CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
          m, n, &alpha, devA, m, devB, m);


  cudaDeviceSynchronize();
  if(ret!=CUBLAS_STATUS_SUCCESS) {
    printf("cublasDtrsm_v2");
    exit(EXIT_FAILURE);
  }
  
  err = cudaMemcpyAsync(B, devB, m*n*sizeof(double), cudaMemcpyDeviceToHost, stream1);

  if(err!=cudaSuccess) {
    printf("cudaMemcpyAsync0");
    exit(EXIT_FAILURE);
  }
  
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2); 
  cublasDestroy(handle);
  cudaFree(devA);
  cudaFree(devB);
}


/**
* cuBLAS matrix invertor wrapper for GNA. Uses cuBLAS linear system solver.
*/
void cuInverseMat(int matSize, double* InMat, double* OutMat) {
  const int blockSize = 16;
  int copyableSize = matSize*matSize*sizeof(double);
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
  cudaMalloc((void**)&devInMat,  copyableSize);
  cudaMalloc((void**)&devOutMat,  copyableSize);

  err = cudaMemcpyAsync(devInMat, InMat, copyableSize, cudaMemcpyHostToDevice, stream1);
  if(err!=cudaSuccess) {
    exit(EXIT_FAILURE);
  }

  GenIdentity<<<dim3(matSize/blockSize + 1, matSize/blockSize + 1), dim3(blockSize, blockSize), 0, stream2>>>(matSize, devOutMat);

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

  err = cudaMemcpyAsync(OutMat, devOutMat, copyableSize, cudaMemcpyDeviceToHost, stream1);

  if(err!=cudaSuccess) {
    exit(EXIT_FAILURE);
  }
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cublasDestroy(handle);
  cudaFree(devInMat);
  cudaFree(devOutMat);
}

/*int main () {
  int m = 20, n = 25;
  double* A = new double[m*m];
  //double* B = new double[k*n];
std::cout << "!!!!!!!!!!!!!!!!!!!!" << std::endl;
  double* C = new double[m*n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        C[i + m*j] = i + j + 1;
        //C[i + n*j] = 1;
      }
      else {
       // if (j < n) A[i + n*j] = 0;
        C[i + m*j] = 0;
      }
    }
  }

  for(int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      if (i == j) A[i + j*m] = 1; else A[i + j*m] = 0;
      std::cout << A[i + j*m] << " ";
    }
    std::cout << std::endl;
  }
std::cout << "BEFORE" <<std::endl;
  cuSolveLowerLS(m, n, A, C);
  //cuInverseMat(n, inM,  outM);
std::cout << "AFTER" << std::endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << C[i + j*m] << " ";
    }
    std::cout << std::endl;
  }
  return 0; 
}
*/
/*  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      A[i+m*j] = 1.0;
      std::cout << A[i+m*j] << " ";
    }
    std::cout << std::endl;
  }

  for(int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      B[i+j*k] = 1.0;
      std::cout << B[i+j*k] << " ";
    }
    std::cout << std::endl;
  }
  cuMultiplyMat(m, n, k, A, B, C);
  
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << C[i+m*j] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
*/
