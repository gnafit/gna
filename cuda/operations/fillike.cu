#include <cuda.h>
#include <iostream>
#include <chrono>

#include "cuElementary.hh"
#include "cuda_config_vars.h"

__global__
void fillike(size_t val, double** ans_array, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;
	ans_array[0][x] = val*1.0;
}

__global__
void fillike(size_t val, float** ans_array, int n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;
	ans_array[0][x] = val*1.0;
}

void cufilllike(size_t val, float** ans_array, int n) {
	fillike<<<n/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>> (val, ans_array, n);
	cudaDeviceSynchronize();
}


void cufilllike(size_t val, double** ans_array, int n) {
	std::cout << "IMHERE!" <<std::endl <<std::endl;
	fillike<<<n/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>> (val, ans_array, n);
	std::cout << "IMHERE!" <<std::endl <<std::endl;
	cudaDeviceSynchronize();
}
