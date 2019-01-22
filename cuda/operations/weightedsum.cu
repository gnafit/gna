#include <cuda.h>
#include <iostream>
#include <chrono>


template <typename T>
__global__ void weightedsum(T** array, T** ans_array, T* weights, int n, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	ans_array[0][x] = array[0][x] * weights[0];
	for (int i = 1; i < n; i++){
		ans_array[0][x] += array[i][x] * weights[i];
	}
}


template<typename T>
__global__ void weightedsumfill(T** array, T** ans_array, T* weights, T k, int n, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	ans_array[0][x] = array[0][x] * weights[0] + k;
	for (int i = 1; i < n; i++){
		ans_array[0][x] += array[i][x] * weights[i];
	}
}

