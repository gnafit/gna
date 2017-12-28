#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "GNAcuGpuArray.hh"
#include "GNAcuGpuMemStates.hh"

template <typename T>
__global__ void vecAdd(T* res, T* inA, T* inB, size_t n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;
	res[x] = inA[x] + inB[x];
}

template <typename T>
__global__ void setByValueGPU(T* res, T val, size_t n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;
	res[x] = val;
}

template <typename T>
GNAcuGpuArray<T>::GNAcuGpuArray() {
	std::cout << "I am created but not inited " << std::endl;
	arrState = NotInitialized;
}

template <typename T>
GNAcuGpuArray<T>::GNAcuGpuArray(T* inArrayPtr, size_t inSize) {
	std::cout << "I am created by ptr " << std::endl;
	cudaSetDevice(0);
	cudaError_t err;
	arrSize = inSize;
	size_t alloc_size = sizeof(T) * inSize;
	err = cudaMalloc((void**)&arrayPtr, alloc_size);
	if (err != cudaSuccess) {
		printf("ERROR: unable to  allocate!\n");
		std::cout << "err is " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = InitializedOnly;
	}
}

template <typename T>
GNAcuGpuArray<T>::~GNAcuGpuArray() {
	cudaFree(arrayPtr);
	cudaDeviceReset();
}

template <typename T>
void GNAcuGpuArray<T>::resize(size_t newSize) {
	cudaError_t err;
	if (arrSize == newSize) return;
	if (arrSize > newSize) {
		arrSize = newSize;
		// TODO: free the end of array
		std::cerr << "WARNING! New array size is less then old size. "
			     "Some data may be lost!"
			  << std::endl;
	} else if (arrSize < newSize) {
		// TODO: resizing without realloc
		cudaFree(arrayPtr);
		size_t alloc_size = sizeof(T) * newSize;
		arrSize = newSize;
		err = cudaMalloc((void**)&arrayPtr, alloc_size);
		if (err != cudaSuccess) {
			printf("ERROR: unable to  allocate!\n");
			std::cerr << "err is " << cudaGetErrorString(err)
				  << std::endl;
			arrState = Crashed;
		}
	}
}

template <typename T>
void GNAcuGpuArray<T>::setByHostArray(T* inHostArr) {
	cudaError_t err;
	err = cudaMemcpy((void**)&arrayPtr, inHostArr, sizeof(T) * arrSize,
			 cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("ERROR: unable to set memory H2D!\n");
		std::cout << "Err is: " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = OnDevice;
	}
}

template <typename T>
void GNAcuGpuArray<T>::setByDeviceArray(T* inDeviceArr) {
	cudaError_t err;
	err = cudaMemcpy(arrayPtr, inDeviceArr, sizeof(T) * arrSize,
			 cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		printf("ERROR: unable to set memory D2D!\n");
		std::cout << "Err is: " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = OnDevice;
	}
}

template <typename T>
void GNAcuGpuArray<T>::setByValue(T value) {
	setByValueGPU<T><<<arrSize, 1>>>(arrayPtr, value, arrSize);
	arrState = OnDevice;
}

template <typename T>
void GNAcuGpuArray<T>::getContentToCPU(T* dst) {
	cudaError_t err;
	double* ttt = new double[arrSize];
	err = cudaMemcpy(dst, arrayPtr, sizeof(T) * arrSize,
			 cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		printf("ERROR: unable to get array values to host!\n");
		std::cout << "Err is: " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = OnHost;
	}

	std::cout << std::endl;
}

template <typename T>
void GNAcuGpuArray<T>::getContent(T* dst) {
	cudaError_t err;
	std::cout << "In getContent: arrSize = " << arrSize << std::endl;
	err = cudaMemcpy(dst, arrayPtr, sizeof(T) * arrSize,
			 cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		printf("ERROR: unable to get array values!\n");
		std::cout << "Err is: " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = OnDevice;
	}
}

template <typename F>
GNAcuGpuArray<F> GNAcuGpuArray<F>::operator+(GNAcuGpuArray<F> rhs) {
	F* resPtr;
	size_t res_size = arrSize;
	if (arrSize != rhs.getArraySize()) {
		std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
			     "smallest will be used!"
			  << std::endl;
		if (arrSize > rhs.getArraySize()) res_size = rhs.getArraySize();
	}
	cudaError_t err;
	err = cudaMalloc((void**)&resPtr, sizeof(F) * res_size);
	if (err != cudaSuccess) {
		printf("ERROR: unable to  allocate memory for add result!\n");
		std::cerr << "err is " << cudaGetErrorString(err) << std::endl;
	}
	vecAdd<F><<<res_size, 1>>>(resPtr, arrayPtr, rhs.getArrayPtr(),
				   res_size);
	F* ttt;
	GNAcuGpuArray<F> res(ttt, res_size);
	res.setByDeviceArray(resPtr);
	res.arrState = OnDevice;
	return res;
}

template <typename T>
GNAcuGpuArray<T>& GNAcuGpuArray<T>::operator=(GNAcuGpuArray<T> rhs) {
// TODO rewrite
	cudaError_t err;
	T* tmpArr;
	//`GNAcuGpuArray<T> res(tmpArr, rhs.getArraySize());
	err = cudaMalloc((void**)&tmpArr, sizeof(T) * rhs.getArraySize());
	rhs.getContent(tmpArr);
	resize(rhs.getArraySize());
	//	arrayPtr = tmpArr;
	err = cudaMemcpy(arrayPtr, tmpArr, sizeof(T) * rhs.getArraySize(),
			 cudaMemcpyDeviceToDevice);

	// setByDeviceArray(tmpArr);

	std::cout << "in Op = : size = " << arrSize;
	//double* tmppp = new double[arrSize];
//	err = cudaMemcpy(tmppp, arrayPtr, sizeof(T) * rhs.getArraySize(),
//			 cudaMemcpyDeviceToHost);
/*	if (err != cudaSuccess) {
		printf("ERROR: unable to  allocate!\n");
		std::cout << "err is " << cudaGetErrorString(err) << std::endl;
		// res.arrState = Crashed;
	}
*/
/*
	getContentToCPU(tmppp);
	for (int i = 0; i < arrSize; i++) {
		std::cout << tmppp[i] << " ";
	}
	std::cout << std::endl;
*/
	return *this;
}

template class GNAcuGpuArray<double>;
template class GNAcuGpuArray<float>;
template class GNAcuGpuArray<int>;
template class GNAcuGpuArray<bool>;
