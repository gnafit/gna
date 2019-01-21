#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


template <typename T> 
__device__ void inverse(T* in, T* out);


template<>
__device__ void inverse <float> (float* in, float* out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (in[idx] != 0) out[idx] =  __frcp_rn(in[idx]); // TODO check if out == 0 by default
}

template <>
__device__ void inverse <double> (double* in, double* out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (in[idx] != 0) out[idx] =  __drcp_rn(in[idx]); // TODO check if out == 0 by default
}

