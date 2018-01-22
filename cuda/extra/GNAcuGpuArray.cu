#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include "GNAcuGpuArray.hh"
#include "GNAcuDataLocation.hh"

template <typename T>
__global__ void vecAdd(T* res, T* inA, T* inB, size_t n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;
	res[x] = inA[x] + inB[x];
}

template <typename T>
__global__ void vecMinus(T* res, T* inA, T* inB, size_t n) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= n) return;
        res[x] = inA[x] - inB[x];
}


template <typename T>
__global__ void vecMult(T* res, T* inA, T* inB, size_t n) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= n) return;
        res[x] = inA[x] * inB[x];
}

template <typename T>
__global__ void vecMultToNum(T* res, T* inA, T inB, size_t n) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= n) return;
        res[x] = inA[x] * inB;
}



template <typename T>
__global__ void setByValueGPU(T* res, T val, size_t n) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= n) return;
	res[x] = val;
}

template <typename T>
__global__ void vecMinusUnar(T* resPtr, T* arrPtr, size_t arrSize) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= arrSize) return;
        resPtr[x] = -arrPtr[x];
}

template <typename T>
GNAcuGpuArray<T>::GNAcuGpuArray() {
	std::cout << "I am created but not inited " << std::endl;
	arrState = NotInitialized;
}

template <typename T>
GNAcuGpuArray<T>::GNAcuGpuArray(T* inArrayPtr, size_t inSize) {
	std::cout << "I am created by ptr " << std::endl;
	cudaError_t err;
	arrSize = inSize;
	size_t alloc_size = sizeof(T) * inSize;
	err = cudaMalloc((void**)&devicePtr, alloc_size);
        std::cout << "Constructor: arrSize is " << arrSize << std::endl;
	if (err != cudaSuccess) {
		printf("ERROR: unable to  allocate!\n");
		std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = InitializedOnly;
	}
}

template <typename T>
DataLocation GNAcuGpuArray<T>::Init(size_t inSize) {
        std::cout << "I am inited by size " << std::endl;
        cudaError_t err;
        arrSize = inSize;
        size_t alloc_size = sizeof(T) * inSize;
        err = cudaMalloc((void**)&devicePtr, alloc_size);
        std::cout << "Constructor: arrSize is " << arrSize << std::endl;
        if (err != cudaSuccess) {
                printf("ERROR: unable to  allocate!\n");
                std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
                arrState = Crashed;
        } else {
                arrState = InitializedOnly;
        }
	return arrState;
}

template <typename T>
GNAcuGpuArray<T>::~GNAcuGpuArray() {
	cudaFree(devicePtr);
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
		cudaFree(devicePtr);
		size_t alloc_size = sizeof(T) * newSize;
		arrSize = newSize;
		err = cudaMalloc((void**)&devicePtr, alloc_size);
                std::cout << "Resize: arrSize is " << arrSize << std::endl;

		if (err != cudaSuccess) {
			printf("ERROR: unable to  allocate!\n");
			std::cerr << "err is " << cudaGetErrorString(err)
				  << std::endl;
			arrState = Crashed;
		}
	}
}

template <typename T>
DataLocation GNAcuGpuArray<T>::setByHostArray(T* inHostArr) {
	cudaError_t err;
std::cout << "In setByHostArray size = " << arrSize << " inHostArr[0] = " << inHostArr[0] <<std::endl;
	err = cudaMemcpy(devicePtr, inHostArr, sizeof(T) * arrSize,
			 cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("ERROR: unable to set memory H2D!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = Device;
	}
	return arrState;
}

template <typename T>
DataLocation GNAcuGpuArray<T>::setByDeviceArray(T* inDeviceArr) {
	cudaError_t err;
	err = cudaMemcpy(devicePtr, inDeviceArr, sizeof(T) * arrSize,
			 cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		printf("ERROR: unable to set memory D2D!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = Device;
	}
        return arrState;
}

template <typename T>
DataLocation GNAcuGpuArray<T>::setByValue(T value) {
	setByValueGPU<T><<<arrSize, 1>>>(devicePtr, value, arrSize);
	arrState = Device;
        return arrState;
}

template <typename T>
DataLocation GNAcuGpuArray<T>::getContentToCPU(T* dst) {
	cudaError_t err;
	err = cudaMemcpy(dst, devicePtr, sizeof(T) * arrSize,
			 cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("ERROR: unable to get array values to host!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = Host;
	}
	return arrState;
}

template <typename T>
DataLocation GNAcuGpuArray<T>::getContent(T* dst) {
	cudaError_t err;
	err = cudaMemcpy(dst, devicePtr, sizeof(T) * arrSize,
			 cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		printf("ERROR: unable to get array values!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
		arrState = Crashed;
	} else {
		arrState = Device;
	}
	return arrState;
}

template <typename T> 
DataLocation GNAcuGpuArray<T>::transferH2D() {
	cudaError_t err;
	if (arrState == NotInitialized) {
		err = cudaMalloc((void**)&devicePtr, arrSize * sizeof(T));
        std::cout << "transfer H2D: arrSize is " << arrSize << std::endl;

        	if (err != cudaSuccess) {
                	printf("ERROR: unable to  allocate!\n");
                	std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
                	arrState = Crashed;
        	}
	}
	err = cudaMemcpy(devicePtr, hostPtr, sizeof(T) * arrSize,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
                printf("ERROR: unable to transfer data H2D!\n");
                std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
                arrState = Crashed;
        } else {
                arrState = Device;
        }
	return arrState;
}

template <typename T>
DataLocation GNAcuGpuArray<T>::transferD2H() {
        cudaError_t err;
        if (arrState == NotInitialized) {
                hostPtr = new T[arrSize];
        }
        err = cudaMemcpy(hostPtr, devicePtr, sizeof(T) * arrSize,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
                printf("ERROR: unable to transfer data D2H!\n");
                std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
                arrState = Crashed;
        } else {
                arrState = Host;
        }
	return arrState;
}


template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator+(GNAcuGpuArray<F> &rhs) {
	if (arrSize != rhs.getArraySize()) {
		std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
			     "smallest will be used!"
			  << std::endl;
	}
	vecAdd<F><<<arrSize, 1>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
				   arrSize);
	arrState = Device;
	return *this;
}

template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator-(GNAcuGpuArray<F> &rhs) {
        if (arrSize != rhs.getArraySize()) {
                std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
                             "smallest will be used!"
                          << std::endl;
        }
        vecMinus<F><<<arrSize, 1>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
                                   arrSize);
        arrState = Device;
        return *this;
}



template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator-() {
	vecMinusUnar<F><<<arrSize, 1>>>(devicePtr, devicePtr, arrSize);
        arrState = Device;
        return *this;
}


template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator*(GNAcuGpuArray<F> &rhs) {
        size_t res_size = arrSize;
        if (arrSize != rhs.getArraySize()) {
                std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
                             "smallest will be used!"
                          << std::endl;
        }
        vecMult<F><<<res_size, 1>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
                                   res_size);
       arrState = Device;
       return *this; 
}


template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator*(F rhs) {
        vecMultToNum<F><<<arrSize, 1>>>(devicePtr, devicePtr, rhs,
                                   arrSize);
        arrState = Device;
        return *this;
}


template <typename T>
GNAcuGpuArray<T> GNAcuGpuArray<T>::operator=(GNAcuGpuArray<T> rhs) {
	GNAcuGpuArray<T> ret(rhs.devicePtr, rhs.arrSize);
//	ret.devicePtr = rhs.devicePtr;
//	ret.arrSize = rhs.arrSize;
        return ret;
}



template class GNAcuGpuArray<double>;
template class GNAcuGpuArray<float>;
template class GNAcuGpuArray<int>;
template class GNAcuGpuArray<bool>;
