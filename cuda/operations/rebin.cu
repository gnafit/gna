#include <cuda.h>
#include <iostream>
#include <chrono>

#include "cuElementary.hh"

#include "cuda_config_vars.h" 
//const int CU_BLOCK_SIZE = 32;
#define GridSize(size) (size/CU_BLOCK_SIZE + 1)



template <typename T>
__global__
void rebin(T** args, T** ints, T** rets, size_t argsize, size_t retsize ) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
//	int y = blockDim.y * blockIdx.y + threadIdx.y;
        //rets[0][x] = 0;
	if (x>=argsize) return;
	for(int i = 0; i < retsize;i++) {
		rets[i][x] = ints[0][argsize*i +x]*args[0][x];
	}

//	ints[0][x] = x;
//	rets[0][x]  =ints[0][x];
	printf("%d, %lf\n", retsize, rets[0][x]);
}

template<typename T>
void curebin(T** args, T** ints, T** rets, size_t argsize, size_t retsize) {
	rebin<<<argsize / CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(args, ints, rets, argsize, retsize);
	cudaDeviceSynchronize();
}

template void curebin<double> (double** args, double** ints, double** rets, size_t argsize, size_t retsize);
