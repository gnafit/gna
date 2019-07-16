#include <cuda.h>
#include <iostream>
#include <chrono>

#include "cuElementary.hh"
#include "cuda_config_vars.h" 

/*
* @brief Element-wise product of N vectors of length M into one
* @return \f$c, c_i = a_i * b_i, i=1..M\f$ 
*
* @author Ilya Lebedev and Anna Fatkina
* @date 2018
*/
template<typename T>
__global__ void d_product(T** array, T** ans_array, unsigned int n, unsigned int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	ans_array[0][x] = array[0][x];
	for (int i = 1; i < n; i++){
		ans_array[0][x] *= array[i][x];
	}
}


/**
* @warning args[0] is vec, args[1] is mat. Mat * vec
*/

template<typename T>
__global__ void d_product_mat2vec(T** args, T** rets, size_t n, size_t m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
        rets[0][x] = 0;
	for(int i = 0; i < n;i++) {
		rets[0][x] += args[1][m*i +x]*args[0][i];
	}

}


/*template<typename T>
__global__ void d_product_mat2mat_EW(T* matL, T* matR, T* rets, size_t cols, size_t rows ) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	rets[x+y*cols] = matL[x+y*cols]  * matR[x+y*cols];
}*/

template <typename T>
void cuproduct(T** array, T** ans_array, unsigned int n, unsigned int m) {
	d_product<<<m/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, n, m);
	cudaDeviceSynchronize();
}


template<typename T> 
void cuproduct_mat2vec(T** args, T** rets, size_t n, size_t m) {
	d_product_mat2vec<<<n / CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(args,rets, n, m);
	cudaDeviceSynchronize();
}


//template __global__ void d_product_mat2mat_EW<double>(double* matL, double* matR, double* rets, size_t cols /* cols in mat*/, size_t rows /* rows in mat*/); 

template void cuproduct<double>(double** array, double** ans_array, unsigned int n, unsigned int m); 
template void cuproduct_mat2vec<double>(double** args, double** rets, size_t n, size_t m);
