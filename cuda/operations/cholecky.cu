#include "cuda_runtime.h"

#include<iostream>
#include<fstream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<chrono>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

/******************************************/
/* SET HERMITIAN POSITIVE DEFINITE MATRIX */
/******************************************/
// --- Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void setPDMatrix(double * h_A, const int N) {
    // --- Initialize random seed
    srand(time(NULL));

    double *h_A_temp = (double *)malloc(N * N * sizeof(double));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_A_temp[i * N + j] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) 
            h_A[i * N + j] = 0.5 * (h_A_temp[i * N + j] + h_A_temp[j * N + i]);

    for (int i = 0; i < N; i++) h_A[i * N + i] = h_A[i * N + i] + N;

}

/************************************/
/* OUTPUT ARRAY FROM CPU */
/************************************/
template <class T>
void outputCPU(const T * h_in, const int M) {

    for (int i = 0; i < M; i++) 
        std::cout << h_in[i] << " ";
    std::cout << std::endl;
}

/************************************/
/* OUTPUT ARRAY FROM GPU */
/************************************/
template <class T>
void outputGPU(const T * d_in, const int M) {

    T *h_in = (T *)malloc(M * sizeof(T));

    cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) 
        std::cout << h_in[i] << " ";
    std::cout << std::endl;
}

void init(int N){
    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // --- CUBLAS initialization
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    /***********************/
    /* SETTING THE PROBLEM */
    /***********************/
    // --- Setting the host, N x N matrix
    double *h_A = (double *)malloc(N * N * sizeof(double));
    setPDMatrix(h_A, N);

    // --- Allocate device space for the input matrix 
    double *d_A; cudaMalloc(&d_A, N * N * sizeof(double));

    // --- Move the relevant matrix from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);

    /****************************************/
    /* COMPUTING THE CHOLESKY DECOMPOSITION */
    /****************************************/
    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo; cudaMalloc(&devInfo, sizeof(int));

    // --- CUDA CHOLESKY initialization
    cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, N, d_A, N, &work_size);

    // --- CUDA POTRF execution
    double *work; cudaMalloc(&work, work_size * sizeof(double));
    cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, N, d_A, N, work, work_size, devInfo);
    int devInfo_h = 0;  cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (devInfo_h != 0) std::cout << "Unsuccessful potrf execution\n\n" << "devInfo = " << devInfo_h << "\n\n";

    // --- At this point, the lower triangular part of A contains the elements of L. 
    /***************************************/
    /* CHECKING THE CHOLESKY DECOMPOSITION */
    /***************************************/
    outputCPU(h_A, N * N);
    outputGPU(d_A, N * N);

    cusolverDnDestroy(solver_handle);
}

int main(int argc, char **argv){
    int N = atoi(argv[1]);
    init(N);
}