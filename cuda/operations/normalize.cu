#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm> 
#include <math.h>

#include"cuda_config_vars.h"


template<typename T>
__device__ double vectorsum(T** array, int start, int limit){
	double del = 0;
	for (int j = start; j < limit; j++)
		del += array[0][j];
	return del;
}

template<typename T> 
__global__ void normalize(T** array, T** ans_array, size_t n) { 
//NOTE: Normalize and normalize_segment can lead to cudaErrorLaunchTimeout 
//      on big amount of elements due to watchdog!
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if (Col < n)
   		ans_array[0][Col] = array[0][Col]/vectorsum(array, 0, n);
}

template<typename T>
__global__ void normalize_segment(double** array, double** ans_array, int* n, int start, int limit) {
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if (Col < n[0])
   		ans_array[0][Col] = array[0][Col]/vectorsum(array, start, limit);
}


template<typename T>
void cunormalize(T** args, T** rets, size_t n) {
	normalize<<<n/CU_BLOCK_SIZE +1, CU_BLOCK_SIZE>>> (args, rets, n);
	cudaDeviceSynchronize();
}

template void cunormalize<double>(double** args, double** rets, size_t n);
