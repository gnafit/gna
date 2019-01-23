#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm> 
#include <math.h>


template<typename T>
__device__ double vectorsum(T** array, int start, int limit){
	double del = 0;
	for (int j = start; j < limit; j++)
		del += array[0][j];
	return del;
}

template<typename T> 
__global__ void normalize(T** array, T** ans_array, int* n) { //NOTE: Normalize and normalize_segment can lead to cudaErrorLaunchTimeout on big amount of elements due to watchdog!
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if (Col < n[0])
   		ans_array[0][Col] = array[0][Col]/vectorsum(array, 0, n[0]);
}

template<typename T>
__global__ void normalize_segment(double** array, double** ans_array, int* n, int start, int limit) {
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if (Col < n[0])
   		ans_array[0][Col] = array[0][Col]/vectorsum(array, start, limit);
}

