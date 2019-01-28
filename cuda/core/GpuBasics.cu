#include "GpuBasics.hh"
#include "cuda.h"

#include <iostream>

template<typename T>
void d_copyH2D(T* dst, T* src, unsigned int N) {
	cudaError_t err;
	cudaMalloc(&dst, N * sizeof(T*));
	err =
		cudaMemcpy(dst, src, N * sizeof(T*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
	}
}

template<typename T>
void d_copyH2D_NOALL(T* dst, T* src, unsigned int N) {
        cudaError_t err;
        err =
                cudaMemcpy(dst, src, N * sizeof(T*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
                std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
        }
}

template <typename T>
void d_cuwr_free(T* ptr) {
	cudaFree(ptr);
}


template <typename T>
 void copyH2D(T* dst, T* src, unsigned int N) {
        d_copyH2D<T>(dst, src, N);
}

template <typename T>
void cuwr_free(T* ptr) {
	d_cuwr_free(ptr);
}

/* void copyH2Dd(double* dst, double* src, unsigned int N) {
        copyH2D<double>(dst, src, N);
}*/


/*template<> void copyH2D<double>(void* dst, void* src, unsigned int N);
template<> void copyH2D_NOALL<double>(void* dst, void* src, unsigned int N) ;
template<> void cuwr_free<double>(void* ptr);

template<> void copyH2D<unsigned int>(void* dst, void* src, unsigned int N);
template<> void copyH2D_NOALL<unsigned int>(void* dst, void* src, unsigned int N) ;
template<> void cuwr_free<unsigned int>(void* ptr);
*/

template void copyH2D<unsigned int>(unsigned int* dst, unsigned int* src, unsigned int N);
template void copyH2D<double>(double* dst, double* src, unsigned int N);
template void cuwr_free<unsigned int>(unsigned int* ptr);
template void cuwr_free<double>(double* ptr);

template void copyH2D<unsigned int*>(unsigned int** dst, unsigned int** src, unsigned int N);
template void copyH2D<double*>(double** dst, double** src, unsigned int N);
template void cuwr_free<unsigned int*>(unsigned int** ptr);
template void cuwr_free<double*>(double** ptr);
