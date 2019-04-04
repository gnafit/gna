#include <cuda.h>
#include <iostream>
#include <chrono>

#include "cuElementary.hh"
#include "cuda_config_vars.h"

__global__
void exp(double** array, double** ans_array, int n, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	for (int i = 0; i < n; i++){
		ans_array[0][x] = exp(array[i][x]);
	}
}

__global__
void exp(float** array, float** ans_array, int n, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	for (int i = 0; i < n; i++){
		ans_array[0][x] = expf(array[i][x]);
}

void cuexp(float** array, float** ans_array, int n, int m) {
	exp<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>> (array, ans_array, n,m);
	cudaDeviceSynchronize();
}


void cuexp(double** array, double** ans_array, int n, int m) {
	exp<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>> (array, ans_array, n,m);
	cudaDeviceSynchronize();
}
