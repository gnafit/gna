#include <cuda.h>
#include <iostream>
#include <chrono>

#include "cuElementary.hh"
#include "cuda_config_vars.h" 

/*
* @brief Element-wise product of N vectors of length M into one
* @return \f$c, c_i = a_i * b_i, i=1..M\f$ 
*
* @author Ilya Lebedev
* @date 2018
*/
template<typename T>
__global__ void d_product(T** array, T** ans_array, unsigned int n, unsigned int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	ans_array[0][x] = array[0][x];
	for (int i = 1; i < n; i++){
		ans_array[0][x] *= array[i][x];
	}
}

template <typename T>
void cuproduct(T** array, T** ans_array, unsigned int n, unsigned int m) {
	cuproduct<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, n, m);
}

template void cuproduct<double>(double** array, double** ans_array, unsigned int n, unsigned int m); 
