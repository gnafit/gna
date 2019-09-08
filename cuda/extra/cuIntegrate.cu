#include "cuIntegrate.hh"
#include "cuda_config_vars.h"
#include <iostream>


//template<typename T>
//__global__ void d_product_mat2mat_EW(T* matL, T* matR, T* rets,           
__global__ void d_product_mat2mat_EW(double* matL, double* matR, double* rets,           
                size_t cols , size_t rows) {

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        rets[x+y*cols] = matL[x+y*cols]  * matR[x+y*cols];
}

template<typename T> 
__global__ void d_integrate2d(T** args, T** ints, T** rets, size_t n, size_t m, size_t k,  size_t m_ordersx_size, size_t m_ordersy_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;
	//if (idx >= n) return;
	printf("idx = %d", idx);
	//rets[idx][idy + m*idz] = 0;
	d_product_mat2mat_EW<<<1,n>>>(args[idx], ints[idx], ints[2*idx + 1], m, k );//ints[2+m*k], m,k);
	//d_product_mat2mat_EW<T>(args[idx], ints[idx], ints[2*idx + 1], m, k );//ints[2+m*k], m,k);
	for (int i = 0; i < m_ordersx_size; i++) {
		for (int j = 0; j < m_ordersy_size; j++) {
			rets[idx][idy + m*idz]  += ints[2*idx+1][m*i*m_ordersx_size +j*m_ordersy_size ];
		}
	}			
	
	// mat of 2d mats

}


template<typename T>
void cuIntegrate2d(T** args, T** ints, T** rets, size_t n, size_t m, size_t  k,  size_t m_ordersx_size, size_t m_ordersy_size) {
  //d_integrate2d<<<dim3(1, 1, 1), 
  d_integrate2d<<<dim3(n/CU_BLOCK_SIZE+1, m/CU_BLOCK_SIZE +1, k/CU_BLOCK_SIZE +1), 
		  dim3(CU_BLOCK_SIZE, CU_BLOCK_SIZE, CU_BLOCK_SIZE)>>>
			(args, ints, rets, n, m,k, m_ordersx_size, m_ordersy_size);
  cudaDeviceSynchronize();
}

template void cuIntegrate2d<double>(double** args, double** ints, double** rets, size_t n, size_t m, size_t  k, size_t m_ordersx_size, size_t m_ordersy_size);
