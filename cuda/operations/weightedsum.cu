#include <cuda.h>
#include <iostream>
#include <chrono>

#include "cuElementary.hh"

#include "cuda_config_vars.h" 

/*
* @brief Weighted sum of N vectors of length M into one
* @return \f$\sum w * x\f$ 
*
* @author Ilya Lebedev
* @date 2018
*/

template <typename T>
__global__ void weightedsum(T** array, T** ans_array, T** weights, unsigned int n, unsigned int m, unsigned int nvars) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
//	if (x >= n) return;
	ans_array[0][x] = array[0][x] * weights[0][0];
	unsigned int i = 1;
	for (; i < nvars; ++i){
		ans_array[0][x] += array[i][x] * weights[0][i];
	}
	for (; i < m; ++i) {
		ans_array[0][x] += array[i][x] ;
	}
}


/*
* @brief Weighted sum of N vectors of length M into one
* @return \f$\sum w * x + k\f$ 
*
* @author Ilya Lebedev
* @date 2018
*/
template<typename T>
__global__ void weightedsumfill(T** array, T** ans_array, T** weights, T k, unsigned int n, unsigned int m, unsigned int nvars) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
//	if (x >= m) return;
	ans_array[0][x] = array[0][x] * weights[0][0] + k;
        unsigned int i = 1;
	for (; i < nvars; i++){
		ans_array[0][x] += array[i][x] * weights[0][i];
	}
	for (; i < m; ++i) {
		ans_array[0][x] += array[i][x] ;
	}
}

template<typename T>
void cuweightedsum(T** array, T** ans_array, T** weights, unsigned int n, unsigned int m, unsigned int nvars) {
	weightedsum<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, weights, n, m, nvars);
	cudaDeviceSynchronize();
}

template<typename T>
void cuweightedsumfill(T** array, T** ans_array, T** weights, T k, unsigned int n, unsigned int m, unsigned int nvars) {
	weightedsumfill<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, weights, k, n, m, nvars);
	cudaDeviceSynchronize();
}

template void cuweightedsum<double>(double** array, double** ans_array, double** weights, unsigned int n, unsigned int m, unsigned int nvars);
template void cuweightedsumfill<double>(double** array, double** ans_array, double** weights, double k, unsigned int n, unsigned int m, unsigned int nvars);
