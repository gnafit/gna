#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm> 
#include <math.h>

#include "cuElementary.hh"
#include "cuda_config_vars.h"


//const int CU_BLOCK_SIZE = 64;
#define GridSize(size) (size/CU_BLOCK_SIZE + 1)

__device__ 
double factorial(double n)
{
    if (n < 2)
        return 1; 
    return n*factorial(n - 1);
}

__device__
void vectorsum(double** array, int n, int i){
	for (int j = 1; j < n; j++)
		array[i/2][0] += array[i/2][j];
	array[i/2][0] *= -2;
}

__global__
void poisson(double** array, double** ans_array, int* n, int amount, int maxn) {
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	
	if(Col < maxn) { //filter unneeded threads
		for (int i = 0; i < amount-1; i+=2){
			//if (n[i]!= n[i+1]) printf("Dimensions do not correspond!\n");
			if (Col < n[i]){
				double fact = factorial(array[i+1][Col]);
				__syncthreads();
    			ans_array[i/2][Col] += log(array[i][Col]) * array[i+1][Col] - array[i][Col] - log(fact);
			if (Col == 0)
	    			vectorsum(ans_array, n[i], i);
    		}
		}
	}
}

__global__
void poissonapprox(double** array, double** ans_array, int* n, int amount, int maxn) {
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	
	if(Col < maxn) { //filter unneeded threads
		for (int i = 0; i < amount-1; i+=2){
			//if (n[i]!= n[i+1]) printf("Dimensions do not correspond!\n");
			if (Col < n[i]){
				__syncthreads();
    			ans_array[i/2][Col] += log(array[i][Col]) * array[i+1][Col] - array[i][Col] - array[i+1][Col]*log(array[i+1][Col]);
			if (Col == 0)
	    			vectorsum(ans_array, n[i], i);
    		}
		}
	}
}

void cupoisson(double** array, double** ans_array, int* n, int amount, int maxn) {
	poisson<<<< amount/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, n, amount, maxn);
	cudaDeviceSynchronize();
}

void cupoissonapprox(double** array, double** ans_array, int* n, int amount, int maxn) {
	poissonapprox<<<< amount/CU_BLOCK_SIZE+1, CU_BLOCK_SIZE>>>(array, ans_array, n, amount, maxn);
	cudaDeviceSynchronize();
}
