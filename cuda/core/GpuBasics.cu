#include "GpuBasics.hh"
#include "cuda.h"

#include <iostream>


template<typename T>
void device_malloc(T* &dst, unsigned int N) {
	cudaMalloc(&dst, N*sizeof(T));
}

template<typename T>
void copyH2D_ALL(T* &dst, T* src, unsigned int N) {
	cudaError_t err;
	cudaMalloc(&dst, N * sizeof(T));
	err =
		cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
	}
}

template<typename T>
void copyH2D_NA(T* dst, T* src, unsigned int N) {
        cudaError_t err;
        err =
                cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
                std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
        }
}



template<typename T>
void copyD2D_NA(T* dst, T* src, unsigned int N) {
        cudaError_t err;
        err =
                cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
                std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
        }
}

template <typename T>
void cuwr_free(T* &ptr) {
	cudaFree(ptr);
    ptr=nullptr;
}

template void copyH2D_ALL<unsigned int>(unsigned int* &dst, unsigned int* src, unsigned int N);
template void copyH2D_ALL<double>(double* &dst, double* src, unsigned int N);
template void cuwr_free<unsigned int>(unsigned int* &ptr);
template void cuwr_free<double>(double* &ptr);

template void copyH2D_ALL<unsigned int*>(unsigned int** &dst, unsigned int** src, unsigned int N);
template void copyH2D_ALL<double*>(double** &dst, double** src, unsigned int N);
template void cuwr_free<unsigned int*>(unsigned int** &ptr);
template void cuwr_free<double*>(double** &ptr);


template void copyH2D_NA<unsigned int*>(unsigned int** dst, unsigned int** src, unsigned int N);
template void copyH2D_NA<double*>(double** dst, double** src, unsigned int N);
template void copyH2D_NA<unsigned int>(unsigned int* dst, unsigned int* src, unsigned int N);
template void copyH2D_NA<double>(double* dst, double* src, unsigned int N);

template void device_malloc<double>(double* &dst, unsigned int N);
template void device_malloc<double*>(double** &dst, unsigned int N);
