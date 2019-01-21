#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


// TODO comple with -fmad=true


/* Inverse template function.
 * Float and double are availible.
 * @return 1/x
 */
template <typename T> 
__device__ void inverse(T* in, T* out);


template<>
__device__ void inverse <float> (float* in, float* out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	out[idx] =  __frcp_rn(in[idx]); // TODO check if out == 0 by default
}

template <>
__device__ void inverse <double> (double* in, double* out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        __drcp_rn(in[idx]); // TODO check if out == 0 by default
}


/* Multiply number k to vector x. 
 * @return k * x
 */
template <typename T>
__device__ void prodNumToVec(T k, T* x, T* res) { 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	res[idx] = k*x[idx];
}
