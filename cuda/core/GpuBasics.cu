#include "GpuBasics.hh"
#include "cuda.h"

#include <iostream>

template<typename T>
void copyH2D(T* dst, T* src, int N) {
	cudaError_t err;
	cudaMalloc(&dst, N * sizeof(T));
	err =
		cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
	}
}

template<typename T>
void copyH2D_NOALL(T* dst, T* src, int N) {
        cudaError_t err;
        err =
                cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
                std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
        }
}

template <typename T>
void cuwr_free(T* ptr) {
	cudaFree(ptr);
}

