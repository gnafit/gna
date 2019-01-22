#include <cuda.h>
#include <iostream>
#include <chrono>


/*
* @brief Summation of N vectors of length M into one
* @return Pointer to array of pointers (should be Rets) with ans_array[0] pointing to result vector
*
* @author Ilya Lebedev
* @date 2018
*/
template <typename T>
__global__ void sum(T** array, T** ans_array, int n, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	ans_array[0][x] = array[0][x];
	for (int i = 1; i < n; i++){
		ans_array[0][x] += array[i][x];
	}
}

