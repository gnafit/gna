#include <cuda.h>
#include <iostream>
#include <chrono>

#include "cuElementary.hh"

#include "cuda_config_vars.h" 
//const int CU_BLOCK_SIZE = 32;
#define GridSize(size) (size/CU_BLOCK_SIZE + 1)

__global__
void selfpower(double** array, double** ans_array, int n, int m, double scale) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	ans_array[0][x] = pow(array[0][x]/scale,array[0][x]/scale);
}

__global__
void selfpower(float** array, float** ans_array, int n, int m, float scale) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	ans_array[0][x] = powf(array[0][x]/scale,array[0][x]/scale);
}

void cuselfpower(double** array, double** ans_array, int n, int m, double scale) {
	selfpower<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, n, m, scale);
	cudaDeviceSynchronize();
}


void cuselfpower(float** array, float** ans_array, int n, int m, float scale) {
	selfpower<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, n, m, scale);
	cudaDeviceSynchronize();
}
