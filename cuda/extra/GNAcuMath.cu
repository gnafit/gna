#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <typeinfo>
#include "GNAcuMath.h"
#include "cublas_v2.h"

/**
  *  Generation of Identity matrix on GPU memory
  */
__global__ void GenIdentity(int n, double* mat) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < n && y < n) mat[x + n * y] = (x == y) ? 1.0 : 0.0;
}

/**
* cuBLAS multiplier wrapper for GNA
*/
void cuMultiplyMat(int m, int n, int k, double* InA, double* InB,
		   double* OutC) {
	cudaSetDevice(0);
	cublasHandle_t handle;
	cublasStatus_t ret;
	cudaError_t err;

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	ret = cublasCreate(&handle);
	if (ret != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR: unable to create cuBLAS handle!\n");
		exit(EXIT_FAILURE);
	}
	double* devA;
	double* devB;
	double* devC;
	cudaMalloc((void**)&devA, m * k * sizeof(double));
	cudaMalloc((void**)&devB, k * n * sizeof(double));
	cudaMalloc((void**)&devC, m * n * sizeof(double));

	cudaMemcpyAsync(devA, InA, m * k * sizeof(double),
			cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(devB, InB, k * n * sizeof(double),
			cudaMemcpyHostToDevice, stream1);
	cudaMemset(devC, 0, m * n * sizeof(double));
	double alpha = 1, beta = 0;
	cudaDeviceSynchronize();
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m,
		    devB, k, &beta, devC, m);

	err = cudaMemcpyAsync(OutC, devC, m * n * sizeof(double),
			      cudaMemcpyDeviceToHost, stream1);

	if (err != cudaSuccess) {
		printf("ERROR: unable to copy memory from device to host! \n");
		exit(EXIT_FAILURE);
	}
	cudaStreamDestroy(stream1);
	cublasDestroy(handle);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
}

/**
* cuBLAS linear system solver wrapper for GNA. A is lower triangular.
*/
void cuSolveLowerLS(int m, int n, double* A, double* B) {
	cublasHandle_t handle;
	cublasStatus_t ret;
	cudaError_t err;
	cudaStream_t stream1;
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

	ret = cublasCreate(&handle);
	if (ret != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR: unable to create cuBLAS handle!\n");
		exit(EXIT_FAILURE);
	}
	double* devA;
	double* devB;
	cudaMalloc((void**)&devA, m * m * sizeof(double));
	cudaMalloc((void**)&devB, m * n * sizeof(double));

	cudaMemcpyAsync(devA, A, m * m * sizeof(double), cudaMemcpyHostToDevice,
			stream1);
	cudaMemcpyAsync(devB, B, m * n * sizeof(double), cudaMemcpyHostToDevice,
			stream1);

	double alpha = 1.0;
	/**
	  *  Solve A*x =  B.
	  */
	ret = cublasDtrsm_v2(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
			     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
			     devA, m, devB, m);

	cudaDeviceSynchronize();
	if (ret != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR: unable to solve linear system with cuBLAS! \n");
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(B, devB, m * n * sizeof(double),
			      cudaMemcpyDeviceToHost, stream1);

	if (err != cudaSuccess) {
		printf("ERROR: unable to copy memory from device to host! \n");
		exit(EXIT_FAILURE);
	}

	cudaStreamDestroy(stream1);
	cublasDestroy(handle);
	cudaFree(devA);
	cudaFree(devB);
}

/**
* cuBLAS matrix invertor wrapper for GNA. Uses cuBLAS linear system solver.
*/
void cuInverseMat(int matSize, double* InMat, double* OutMat) {
	const int blockSize = 16;
	int copyableSize = matSize * matSize * sizeof(double);
	cudaSetDevice(0);
	cublasHandle_t handle;
	cublasStatus_t ret;
	cudaError_t err;

	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	ret = cublasCreate(&handle);
	if (ret != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR: unable to create cuBLAS handle!\n");
		exit(EXIT_FAILURE);
	}

	double* devInMat;
	double* devOutMat;
	cudaMalloc((void**)&devInMat, copyableSize);
	cudaMalloc((void**)&devOutMat, copyableSize);

	err = cudaMemcpyAsync(devInMat, InMat, copyableSize,
			      cudaMemcpyHostToDevice, stream1);
	if (err != cudaSuccess) {
		printf("ERROR: unable to copy memory from host to device! \n");
		exit(EXIT_FAILURE);
	}

	GenIdentity<<<dim3(matSize / blockSize + 1, matSize / blockSize + 1),
		      dim3(blockSize, blockSize), 0, stream2>>>(matSize,
								devOutMat);

	double alpha = 1.0;
	/**
	  *  Solve A*x = alpha * B to invert matrix. In this case B is Identity,
	 * alpha == 1.
	  */
	ret =
	    cublasDtrsm_v2(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
			   CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, matSize, matSize,
			   &alpha, devInMat, matSize, devOutMat, matSize);

	cudaDeviceSynchronize();
	if (ret != CUBLAS_STATUS_SUCCESS) {
		printf("ERROR: unable to invert matrix with cuBLAS! \n");
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(OutMat, devOutMat, copyableSize,
			      cudaMemcpyDeviceToHost, stream1);

	if (err != cudaSuccess) {
		printf("ERROR: unable to copy memory from device to host! \n");
		exit(EXIT_FAILURE);
	}
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cublasDestroy(handle);
	cudaFree(devInMat);
	cudaFree(devOutMat);
	cudaDeviceReset();
}
