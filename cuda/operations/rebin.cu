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
        rets[0][x] = 0;
	for(int i = 0; i < argsize;i++) {
		rets[0][x] += ints[0][retsize*i +x]*args[0][i];
	}

}

template<typename T>
void curebin(T** args, T** ints, T** rets, size_t argsize, size_t retsize) {
	rebin<<<retsize / CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(args, ints, rets, argsize, retsize);
	cudaDeviceSynchronize();
}

template void curebin<double> (double** args, double** ints, double** rets, size_t argsize, size_t retsize);
