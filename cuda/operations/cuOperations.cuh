#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


// TODO comple with -fmad=true


template<typename T>
__global__ void d_product_mat2mat_EW(T* matL, T* matR, T** rets, size_t cols /* cols in mat*/, size_t rows /* rows in mat*/);

/* Inverse template function.
 * Float and double are availible.
 * @return 1/x
 */
template <typename T> 
__device__ void inverse(T* in, T* out, unsigned int m);


template<>
__device__ void inverse <float> (float* in, float* out, unsigned int m) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m)
	  out[idx] =  __frcp_rn(in[idx]); // TODO check if out == 0 by default
}

template <>
__device__ void inverse <double> (double* in, double* out, unsigned int m) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m)
          //out[idx] = __drcp_rn(in[idx]); // TODO check if out == 0 by default
	  out[idx] = 1.0/in[idx];
}
/*
template <>
__device__ void inverse <size_t> (size_t* in, size_t* out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        out[idx] =in[idx]; // TODO check if out == 0 by default
}
*/

/* Multiply number k to vector x. 
 * @return k * x
 */
template <typename T>
__device__ void prodNumToVec(T k, T* x, T* res) { 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	res[idx] = k*x[idx];
}


/* 
 * Sin of k*x
 *
 */

template <typename T>
__device__ void arr_sin (T k, T* x, T* out);

template <>
__device__ void arr_sin (double k, double* x, double* out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
	out[idx] = sin(k*x[idx]);
} 

template <>
__device__ void arr_sin (float k, float* x, float* out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
	out[idx] = sinf(k*x[idx]);
} 

/*
 * Multiply myself to sin(k*x)
 */

template <typename T>
__device__ void mult_by_arr_sin (T k, T* x, T* me);

template <>
__device__ void mult_by_arr_sin (float k, float* x, float* me) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        me[idx] = me[idx] * sinf(k*x[idx]);
}


template <>
__device__ void mult_by_arr_sin (double k, double* x, double* me) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        me[idx] = me[idx] * sin(k*x[idx]);
}

