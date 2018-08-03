#include "GpuBasics.hh"
#include "cuda.h"

#include <iostream>

void copyH2D(double** dst, double** src, int N) {
	cudaError_t err;
	err =
		cudaMemcpy(dst, src, N * sizeof(double*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
	}
}
