#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm> 
#include <math.h>


template<typename T>
__device__ double vectorsumcolumn(T** array, int n, int x){
	double del = 0;
	for (int j = 0; j < n; j++){
		del += array[0][j*n + x];
	}
	return del==0.0 ? 1.0 : del;
}

template <typename T>
__global__ void renormalizediag(T** array, T** ans_array, int* n, int* m, int Diag, int multiplier) { 
//NOTE: Normalize and normalize_segment can lead to cudaErrorLaunchTimeout on big amount of elements due to watchdog!
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if (x < n[0] & y < n[0]){
		ans_array[0][y*n[0]+x] = array[0][y*n[0]+x];
		if (abs(x - y) <= Diag - 1) //multiply diagonals
			ans_array[0][y*n[0] + x] *= multiplier;
		__syncthreads();
   		ans_array[0][y*n[0] + x] = ans_array[0][y*n[0] + x]/vectorsumcolumn(ans_array, n[0], x);
	}
