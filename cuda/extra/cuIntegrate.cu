#include "cuIntegrate.hh"
#include "cuda_config_vars.h"
#include <iostream>


template<typename T>
__global__ void d_product_mat2mat_EW(T* matL, T* matR, T* rets,           
                size_t cols , size_t rows) {

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        rets[x+y*cols] = matL[x+y*cols]  * matR[x+y*cols];
}

template<typename T> 
__global__ void d_integrate2d(T** args, T** ints, T** rets, size_t n, size_t m, size_t k) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;
	// ints[0] = xorders
	// ints[1] = yorders
	rets[idx][idy + m*idz] = 0;
	d_product_mat2mat_EW<T><<<1,n>>>(args[idx], ints[idx], ints[2*idx], m, k );//ints[2+m*k], m,k);
	for (int i = 0; i < (size_t) ints[0][0]; i++) {
		for (int j = 0; j < (size_t)ints[0][1]; j++) {
			rets[idx][idy + m*idz]  += ints[2*idx][m*i*(size_t)ints[0][0] +j*(size_t)ints[0][1] ];
		}
	}			
	
	// mat of 2d mats

}


template<typename T>
void cuIntegrate2d(T** args, T** ints, T** rets, size_t n, size_t m, size_t  k) {
  d_integrate2d<<<dim3(n/CU_BLOCK_SIZE+1, m/CU_BLOCK_SIZE +1, k/CU_BLOCK_SIZE +1), 
		  dim3(CU_BLOCK_SIZE,CU_BLOCK_SIZE,CU_BLOCK_SIZE)>>>
			(args, ints, rets, n, m,k);
  cudaDeviceSynchronize();
}

template void cuIntegrate2d<double>(double** args, double** ints, double** rets, size_t n, size_t m, size_t  k);
