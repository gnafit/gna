#include <cuda.h>
#include <iostream>
#include <chrono>
#include "cuElementary.hh"

#include "cuda_config_vars.h"

/*
* @brief Summation of N vectors of length M into one
* @return Pointer to array of pointers (should be Rets) with ans_array[0] pointing to result vector
*
* @author Ilya Lebedev
* @date 2018
*/
template <typename T>
__global__ void sum(T** array, T** ans_array, unsigned int n, unsigned int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	ans_array[0][x] = array[0][x];
	for (int i = 1; i < n; i++){
		ans_array[0][x] += array[i][x];
	}
}

template <typename T>
void cusum(T** array, T** ans_array, unsigned int n, unsigned int m) {
	sum<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, n, m);
	cudaDeviceSynchronize();
}

template void cusum<double>(double** array, double** ans_array, unsigned int n, unsigned int m);
template void cusum<float>(float** array, float** ans_array, unsigned int n, unsigned int m);
