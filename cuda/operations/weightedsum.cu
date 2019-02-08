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
__global__ void weightedsum(T** array, T** ans_array, T** weights, unsigned int n, unsigned int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;
	ans_array[0][x] = array[0][x] * weights[0][0];
	printf("WEIGHTS ==================== %lf \n", weights[0][0] );
//	ans_array[0][x] = 1;
//	printf("DEBUGOUT I M ON GPU\n");
	for (unsigned int i = 1; i < m; ++i){
		ans_array[0][x] = ans_array[0][x] + array[i][x] * weights[0][i];
        printf("WEIGHTS NEXT ==================== %lf \n", weights[0][i] );

//		ans_array[0][x] += array[i][x] ;
	}

	printf("ANS = %lf \n", ans_array[0][x]);
}


/*
* @brief Weighted sum of N vectors of length M into one
* @return \f$\sum w * x + k\f$ 
*
* @author Ilya Lebedev
* @date 2018
*/
template<typename T>
__global__ void weightedsumfill(T** array, T** ans_array, T* weights, T k, int n, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	ans_array[0][x] = array[0][x] * weights[0] + k;
	for (int i = 1; i < n; i++){
		ans_array[0][x] += array[i][x] * weights[i];
	}
}

template<typename T>
void cuweightedsum(T** array, T** ans_array, T** weights, unsigned int n, unsigned int m) {
	weightedsum<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, weights, n, m);
	std::cout << "I AM I AM I AM I AM!!!!!!!" << std::endl;
	cudaDeviceSynchronize();
}

template void cuweightedsum<double>(double** array, double** ans_array, double** weights, unsigned int n, unsigned int m);
