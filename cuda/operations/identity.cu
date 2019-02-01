#include "cuElementary.hh"

#include "cuda_config_vars.h"

template<typename T>
__global__ void d_identity(T** in, T** out, unsigned int N, unsigned int M) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;	
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < N && y < M) out[x][y] = in[x][y];	
	
}

template<typename T>
void identity_gpu(T** in, T** out, unsigned int N, unsigned int M) {
	d_identity<<<dim3(M/CU_BLOCK_SIZE+1, N/CU_BLOCK_SIZE+1), 
		     dim3(CU_BLOCK_SIZE, CU_BLOCK_SIZE)>>> (in, out, N, M);
	cudaDeviceSynchronize();
}


template void identity_gpu<double>(double** in, double** out, unsigned int n, unsigned int m );
