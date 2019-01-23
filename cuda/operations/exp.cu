#include <cuda.h>
#include <iostream>
#include <chrono>

__global__
void exp(double** array, double** ans_array, int n, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	for (int i = 0; i < n; i++){
		ans_array[0][x] = expf(array[i][x]);
	}
}

