#include "GpuBasics.hh"
#include "cuda.h"
#include "cstddef"

#include <iostream>


template<typename T>
void debug_drop(T* in, unsigned int N) {
	T* tmp = new T[N];
	cudaMemcpy(tmp, in, N * sizeof(T), cudaMemcpyDeviceToHost);
	std::cout << "Debug drop:" << std::endl;
	for (unsigned int i = 0; i < N; ++i) {
		std::cout << tmp[i] << " ";
	}
	std::cout << std::endl;
}

template<typename T>
void debug_drop(T** in, size_t M /*how many arrs*/ , size_t N /*length of single arr*/ ) {
	std::cout << "Debug multidim drop:" << std::endl;
	T** bigtmp = new T*[M];
	cudaMemcpy(bigtmp, in, M*sizeof(T*), cudaMemcpyDeviceToHost);
	for(int j = 0; j < M ; j++) {
		T* tmp = new T[N];
		cudaMemcpy(tmp, bigtmp[j], N * sizeof(T), cudaMemcpyDeviceToHost);
		for (unsigned int i = 0; i < N; ++i) {
			std::cout << tmp[i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout <<  std::endl;
}

template<typename T>
void device_malloc(T* &dst, unsigned int N) {
	cudaError_t err = cudaMalloc(&dst, N*sizeof(T));
	if (err != cudaSuccess) {
		std::cerr << "Allocation err is " << cudaGetErrorString(err) << std::endl;
	}
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
//    ptr=nullptr;
}

template void copyH2D_ALL<unsigned int>(unsigned int* &dst, unsigned int* src, unsigned int N);
template void copyH2D_ALL<size_t>(size_t* &dst, size_t* src, unsigned int N);
template void copyH2D_ALL<double>(double* &dst, double* src, unsigned int N);
template void cuwr_free<unsigned int>(unsigned int* &ptr);
template void cuwr_free<size_t>(size_t* &ptr);
template void cuwr_free<double>(double* &ptr);

template void copyH2D_ALL<unsigned int*>(unsigned int** &dst, unsigned int** src, unsigned int N);
template void copyH2D_ALL<size_t*>(size_t** &dst, size_t** src, unsigned int N);
template void copyH2D_ALL<double*>(double** &dst, double** src, unsigned int N);
template void cuwr_free<unsigned int*>(unsigned int** &ptr);
template void cuwr_free<size_t*>(size_t** &ptr);
template void cuwr_free<double*>(double** &ptr);


template void copyH2D_NA<unsigned int*>(unsigned int** dst, unsigned int** src, unsigned int N);
template void copyH2D_NA<double*>(double** dst, double** src, unsigned int N);
template void copyH2D_NA<unsigned int>(unsigned int* dst, unsigned int* src, unsigned int N);
template void copyH2D_NA<double>(double* dst, double* src, unsigned int N);

template void device_malloc<double>(double* &dst, unsigned int N);
template void device_malloc<double*>(double** &dst, unsigned int N);

template void debug_drop<double>(double* dst, unsigned int N);
template void debug_drop<double*>(double** dst, unsigned int N);
template void debug_drop<double>(double** in, size_t M /*how many arrs*/ , size_t N /*length of single arr*/ ) ;
