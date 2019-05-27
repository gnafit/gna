#include <cuda.h>
#include <iostream>
#include <chrono>

#include "cuElementary.hh"

#include "cuda_config_vars.h" 

/*
* @brief Weighted sum of N vectors of length M into one
* @return \f$\sum w * x\f$ 
*
* @author Ilya Lebedev and Anna Fatkina
* @date 2018
*/

template <typename T>
__global__ void weightedsum(T** array, T** ans_array, T** weights, unsigned int n, unsigned int m, unsigned int nvars) {
	int x = blockDim.x * blockIdx.x + threadIdx.x; /* num of element in array */
	if (x >= n) return;
//	ans_array[0][x] = array[0][x];
	ans_array[0][x] = array[0][x] * weights[0][0];
	unsigned int i = 1;
	for (; i < nvars; ++i){
		ans_array[0][x] += array[i][x] * weights[i][0];
	}

/*	printf("BAD FUNC");
	for (int i = 0; i < nvars; i++) {
		printf("%lf %lf \n", array[0][i], array[i][0]);
	}
*/
}


/*
* @brief Weighted sum of N vectors of length M into one
* @return \f$\sum w * x + k\f$ 
*
* @author Ilya Lebedev and Anna Fatkina
* @date 2018
*/
template<typename T>
__global__ void weightedsumfill(T** array, T** ans_array, T** weights, T k, unsigned int n, unsigned int m, unsigned int nvars) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;
	ans_array[0][x] = array[0][x] * weights[0][0] + k;
        unsigned int i = 1;
	for (; i < nvars; i++){
		ans_array[0][x] += array[i][x] * weights[i][0];
	}
/*	for (; i < m; ++i) {
		ans_array[0][x] += array[i][x] ;
	}
*/
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
