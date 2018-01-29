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
__global__ void vecMult(T* res, T* inA, T inB, size_t n) {
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
__global__ void vecMinusUnar(T* arrPtr, size_t arrSize) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
        if (x >= arrSize) return;
        arrPtr[x] = -arrPtr[x];
}

template <typename T>
GNAcuGpuArray<T>::GNAcuGpuArray() {
#ifdef CU_DEBUG
	std::cout << "GPUArray is created but not inited " << std::endl;
#endif
	arrState = NotInitialized;
}

template <typename T>
GNAcuGpuArray<T>::GNAcuGpuArray(size_t inSize) {
#ifdef CU_DEBUG
	std::cout << "GPU Array is created by size (constructor)" << std::endl;
#endif
	cudaError_t err;
	arrSize = inSize;
	size_t alloc_size = sizeof(T) * inSize;
	err = cudaMalloc((void**)&devicePtr, alloc_size);
/*
  std::chrono::time_point<std::chrono::system_clock> start, end;
  end = std::chrono::system_clock::now();
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
std::cout << "After Malloc Constructor: " << std::ctime(&end_time) << std::endl;
*/
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to  allocate!\n");
		std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
#endif
		arrState = Crashed;
	} else {
		arrState = InitializedOnly;
	}
}

template <typename T>
DataLocation GNAcuGpuArray<T>::Init(size_t inSize) {
#ifdef CU_DEBUG
        std::cout << "GPU Array is inited by size (Init)" << std::endl;
#endif
        cudaError_t err;
        arrSize = inSize;
        size_t alloc_size = sizeof(T) * inSize;
        err = cudaMalloc((void**)&devicePtr, alloc_size);
/*
std::chrono::time_point<std::chrono::system_clock> start, end;
  end = std::chrono::system_clock::now();
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
std::cout << "After Malloc Init: " << std::ctime(&end_time) << std::endl;
*/
        if (err != cudaSuccess) {
#ifdef CU_DEBUG
                printf("ERROR: unable to  allocate!\n");
                std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
#endif
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
	err = cudaMemcpy(devicePtr, inHostArr, sizeof(T) * arrSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to copy memory H2D!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
		arrState = Crashed;
	} else {
		arrState = Device;
	}
	return arrState;
}

template <typename T>
DataLocation GNAcuGpuArray<T>::setByDeviceArray(T* inDeviceArr) {
	cudaError_t err;
	err = cudaMemcpy(devicePtr, inDeviceArr, sizeof(T) * arrSize, cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to copy memory D2D!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
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
	err = cudaMemcpy(dst, devicePtr, sizeof(T) * arrSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to get array values to host!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
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
#ifdef CU_DEBUG
		printf("ERROR: unable to get array values!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
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
        	if (err != cudaSuccess) {
#ifdef CU_DEBUG
                	printf("ERROR: unable to  allocate!\n");
                	std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
#endif
                	arrState = Crashed;
        	}
	}
	err = cudaMemcpy(devicePtr, hostPtr, sizeof(T) * arrSize,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
#ifdef CU_DEBUG
                printf("ERROR: unable to transfer data H2D!\n");
                std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
                arrState = Crashed;
        } else {
                arrState = Device;
        }
	return arrState;
}

template <typename T>
void GNAcuGpuArray<T>::transferD2H() {
        cudaError_t err;
        hostPtr = new T[arrSize];
        err = cudaMemcpy(hostPtr, devicePtr, sizeof(T) * arrSize,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
#ifdef CU_DEBUG
                printf("ERROR: unable to transfer data D2H!\n");
                std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
                arrState = Crashed;
        } else {
                arrState = Host;
        }
}


template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator+=(GNAcuGpuArray<F> &rhs) {
	int smallest_size = arrSize;
	if (arrSize != rhs.getArraySize()) {
#ifdef CU_DEBUG
		if(arrSize > rhs.getArraySize()) { smallest_size = rhs.getArraySize(); }
		std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
			     "result may be not valid!"
			  << std::endl;
#endif
	}
	vecAdd<F><<<smallest_size, 1>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
				   smallest_size);
	arrState = Device;
	return *this;
}

template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator-=(GNAcuGpuArray<F> &rhs) {
	int smallest_size = arrSize;
        if (arrSize != rhs.getArraySize()) {
                if(arrSize > rhs.getArraySize()) { smallest_size = rhs.getArraySize(); }
#ifdef CU_DEBUG
                std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
                             "result may be not valid!"
                          << std::endl;
#endif
        }
        vecMinus<F><<<smallest_size, 1>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
                                    smallest_size);
        arrState = Device;
        return *this;
}



template <typename F>
void GNAcuGpuArray<F>::negate() {
	vecMinusUnar<F><<<arrSize, 1>>>(devicePtr, arrSize);
        arrState = Device;
}


template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator*=(GNAcuGpuArray<F> &rhs) {
        size_t res_size = arrSize;
        if (arrSize != rhs.getArraySize()) {
		if(arrSize > rhs.getArraySize()) {res_size = rhs.getArraySize(); }
#ifdef CU_DEBUG 
                std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
                             "result may be not valid!"
                          << std::endl;
#endif
        }
        vecMult<F><<<res_size, 1>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
                                   res_size);
       arrState = Device;
       return *this; 
}


template <typename F>
GNAcuGpuArray<F>& GNAcuGpuArray<F>::operator*=(F rhs) {
        vecMult<F><<<arrSize, 1>>>(devicePtr, devicePtr, rhs,
                                   arrSize);
        arrState = Device;
        return *this;
}


template <typename T>
GNAcuGpuArray<T> GNAcuGpuArray<T>::operator=(GNAcuGpuArray<T> rhs) {
	GNAcuGpuArray<T> ret(rhs.arrSize);
	ret.setByDeviceArray(rhs.getArrayPtr());
        return ret;
}

template <typename T>
void GNAcuGpuArray<T>::dump() {
	T* tmp = new T[arrSize];
	getContentToCPU(tmp);
	for (int i = 0; i < arrSize; i++) {
		std::cout << tmp[i] << " ";
        }
	std::cout << std::endl;
}

template class GNAcuGpuArray<double>;
//template class GNAcuGpuArray<float>;
//template class GNAcuGpuArray<int>;
//template class GNAcuGpuArray<bool>;
