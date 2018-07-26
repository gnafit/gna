#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include "GpuArray.hh"
#include "DataLocation.hh"

#define GridSize(size) (size/CU_BLOCK_SIZE + 1)



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
GpuArray<T>::GpuArray(T* inHostPtr) {
#ifdef CU_DEBUG
	std::cout << "GPUArray is created but not inited " << std::endl;
#endif
	hostPtr = inHostPtr;
	if(inHostPtr == nullptr)    dataLoc = DataLocation::NoData;
	else 			    dataLoc = DataLocation::Host;
	syncFlag = SyncFlag::Unsynchronized;
}

template <typename T>
GpuArray<T>::GpuArray(size_t inSize, T* inHostPtr) {
        hostPtr = inHostPtr;
	if(inHostPtr == nullptr)    dataLoc = DataLocation::NoData;
	syncFlag = SyncFlag::Unsynchronized;
#ifdef CU_DEBUG
	std::cout << "GPU Array is created by size (constructor)" << std::endl;
#endif
	cudaError_t err;
	arrSize = inSize;
	size_t alloc_size = sizeof(T) * inSize;
	err = cudaMalloc((void**)&devicePtr, alloc_size);
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to  allocate!\n");
		std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
#endif
		dataLoc = DataLocation::Crashed;
	} else {
		deviceMemAllocated = true;
		dataLoc = DataLocation::InitializedOnly;
	}
}

template <typename T>
DataLocation GpuArray<T>::Init(size_t inSize, T* inHostPtr) {
        hostPtr = inHostPtr;
	if(inHostPtr == nullptr)    dataLoc = DataLocation::NoData;
	syncFlag = SyncFlag::Unsynchronized;
#ifdef CU_DEBUG
        std::cout << "GPU Array is inited by size (Init)" << std::endl;
#endif
        cudaError_t err;
        arrSize = inSize;
        size_t alloc_size = sizeof(T) * inSize;
        err = cudaMalloc((void**)&devicePtr, alloc_size);
        if (err != cudaSuccess) {
#ifdef CU_DEBUG
                printf("ERROR: unable to  allocate!\n");
                std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
#endif
                dataLoc = DataLocation::Crashed;
        } else {
		deviceMemAllocated = true; 
                dataLoc = DataLocation::InitializedOnly;
        }
	return dataLoc;
}

template <typename T>
GpuArray<T>::~GpuArray() {
	cudaFree(devicePtr);
}

template <typename T>
DataLocation GpuArray<T>::setByHostArray(T* inHostArr) {
	cudaError_t err;
	err = cudaMemcpy(devicePtr, inHostArr, sizeof(T) * arrSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to copy memory H2D!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
		dataLoc = DataLocation::Crashed;
	} else {
		dataLoc = DataLocation::Device;
	}
	return dataLoc;
}

template <typename T>
DataLocation GpuArray<T>::setByDeviceArray(T* inDeviceArr) {
	cudaError_t err;
	err = cudaMemcpy(devicePtr, inDeviceArr, sizeof(T) * arrSize, cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to copy memory D2D!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
		dataLoc = DataLocation::Crashed;
	} else {
		dataLoc = DataLocation::Device;
	}
        return dataLoc;
}

template <typename T>
DataLocation GpuArray<T>::setByValue(T value) {
	setByValueGPU<T><<<GridSize(arrSize), CU_BLOCK_SIZE>>>(devicePtr, value, arrSize);
	dataLoc = DataLocation::Device;
        return dataLoc;
}

template <typename T>
DataLocation GpuArray<T>::getContentToCPU(T* dst) {
	cudaError_t err;
	err = cudaMemcpy(dst, devicePtr, sizeof(T) * arrSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to get array values to host!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
		dataLoc = DataLocation::Crashed;
	} else {
		dataLoc = DataLocation::Host;
	}
	return dataLoc;
}

template <typename T>
DataLocation GpuArray<T>::getContent(T* dst) {
	cudaError_t err;
	err = cudaMemcpy(dst, devicePtr, sizeof(T) * arrSize,
			 cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
#ifdef CU_DEBUG
		printf("ERROR: unable to get array values!\n");
		std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
		dataLoc = DataLocation::Crashed;
	} else {
		dataLoc = DataLocation::Device;
	}
	return dataLoc;
}

template <typename T> 
void GpuArray<T>::sync_H2D() {
#ifdef CU_DEBUG_3
    	printf("Sync to H2D\n");
#endif
	cudaError_t err;
	if (dataLoc == DataLocation::NotInitialized) {
		err = cudaMalloc((void**)&devicePtr, arrSize * sizeof(T));
        	if (err != cudaSuccess) {
#ifdef CU_DEBUG
                	printf("ERROR: unable to  allocate!\n");
                	std::cerr << "Err is " << cudaGetErrorString(err) << std::endl;
#endif
                	dataLoc = DataLocation::Crashed;
			return;
        	}
		deviceMemAllocated = true;
	}
	err = cudaMemcpy(devicePtr, hostPtr, sizeof(T) * arrSize,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
#ifdef CU_DEBUG
                printf("ERROR: unable to transfer data H2D!\n");
                std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
                dataLoc = DataLocation::Crashed;
		syncFlag = SyncFlag::SyncFailed;
        } else {
                dataLoc = DataLocation::Device;
                syncFlag = SyncFlag::Synchronized;
        }
}

template <typename T>
void GpuArray<T>::sync_D2H() {
#ifdef CU_DEBUG_3
	printf("Sync D2H\n");
#endif
        cudaError_t err;
        err = cudaMemcpy(hostPtr, devicePtr, sizeof(T) * arrSize,
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
#ifdef CU_DEBUG
                printf("ERROR: unable to transfer data D2H!\n");
                std::cerr << "Err is: " << cudaGetErrorString(err) << std::endl;
#endif
                dataLoc = DataLocation::Crashed;
		syncFlag = SyncFlag::SyncFailed;
        } else {
                dataLoc = DataLocation::Host;
		syncFlag = SyncFlag::Synchronized;
        }
}


template <typename T>
void GpuArray<T>::sync(DataLocation loc) {
/**
Copies the actual data to the loc location
*/
  if (dataLoc == loc || syncFlag == SyncFlag::Synchronized) {
#ifdef CU_DEBUG_2
    std::cerr << "Relevant data on "<< loc << "  -- no synchronization needed" << std::endl;
#endif
  } else if((dataLoc == DataLocation::Device && loc == DataLocation::Host)) {
    sync_D2H();
  } else if((dataLoc == DataLocation::Host && loc == DataLocation::Device)) {
    sync_H2D();
  } else if (dataLoc == DataLocation::NoData) {
    throw std::runtime_error("Data is not initialized");
  } else {
    syncFlag = SyncFlag::SyncFailed;
#ifdef CU_DEBUG_2
    std::cerr << "Cannot be synchronized! Smth wrong: current location state is <" << dataLoc << ">, new data location state is <" << loc << ">" << std::endl;
#endif
  }
}


template <typename T>
void GpuArray<T>::synchronize() {
/**
Makes data the same on GPU and CPU
*/
  if (dataLoc == DataLocation::Device) {
    sync(DataLocation::Host);
    syncFlag = SyncFlag::Synchronized;
#ifdef CU_DEBUG_3
    printf("Sync to GPU\n");
#endif
  } else if(dataLoc == DataLocation::Host) {
    sync(DataLocation::Device);
    syncFlag = SyncFlag::Synchronized;
#ifdef CU_DEBUG_3
    printf("Sync to CPU\n");
#endif
  } else {
#ifdef CU_DEBUG_2
    std::cerr << "Unable to sync data as current GPU memory state is " <<  dataLoc << std::endl;
#endif
    syncFlag = SyncFlag::SyncFailed;
  }
}


template <typename F>
GpuArray<F>& GpuArray<F>::operator+=(GpuArray<F> &rhs) {
	int smallest_size = arrSize;
	if (arrSize != rhs.getArraySize()) {
#ifdef CU_DEBUG
		if(arrSize > rhs.getArraySize()) { smallest_size = rhs.getArraySize(); }
		std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
			     "result may be not valid!"
			  << std::endl;
#endif
	}
	vecAdd<F><<<GridSize(smallest_size), CU_BLOCK_SIZE>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
				   smallest_size);
	dataLoc = DataLocation::Device;
	return *this;
}

template <typename F>
GpuArray<F>& GpuArray<F>::operator-=(GpuArray<F> &rhs) {
	int smallest_size = arrSize;
        if (arrSize != rhs.getArraySize()) {
                if(arrSize > rhs.getArraySize()) { smallest_size = rhs.getArraySize(); }
#ifdef CU_DEBUG
                std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
                             "result may be not valid!"
                          << std::endl;
#endif
        }
        vecMinus<F><<<GridSize(smallest_size), CU_BLOCK_SIZE>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
                                    smallest_size);
        dataLoc = DataLocation::Device;
        return *this;
}



template <typename F>
void GpuArray<F>::negate() {
	vecMinusUnar<F><<<GridSize(arrSize), CU_BLOCK_SIZE>>>(devicePtr, arrSize);
        dataLoc = DataLocation::Device;
}


template <typename F>
GpuArray<F>& GpuArray<F>::operator*=(GpuArray<F> &rhs) {
        size_t res_size = arrSize;
        if (arrSize != rhs.getArraySize()) {
		if(arrSize > rhs.getArraySize()) {res_size = rhs.getArraySize(); }
#ifdef CU_DEBUG 
                std::cerr << "ERROR: Sizes of lhs and rhs are different! The "
                             "result may be not valid!"
                          << std::endl;
#endif
        }
        vecMult<F><<<GridSize(res_size), CU_BLOCK_SIZE>>>(devicePtr, devicePtr, rhs.getArrayPtr(),
                                   res_size);
       dataLoc = DataLocation::Device;
       return *this; 
}


template <typename F>
GpuArray<F>& GpuArray<F>::operator*=(F rhs) {
        vecMult<F><<<GridSize(arrSize), CU_BLOCK_SIZE>>>(devicePtr, devicePtr, rhs,
                                   arrSize);
        dataLoc = DataLocation::Device;
        return *this;
}


template <typename T>
GpuArray<T> GpuArray<T>::operator=(GpuArray<T> rhs) {
	GpuArray<T> ret(rhs.arrSize);
	ret.setByDeviceArray(rhs.getArrayPtr());
        return ret;
}

template <typename T>
void GpuArray<T>::dump() {
	T* tmp = new T[arrSize];
	getContentToCPU(tmp);
	for (int i = 0; i < arrSize; i++) {
		std::cout << tmp[i] << " ";
        }
	std::cout << std::endl;
}

template class GpuArray<double>;
//template class GpuArray<float>;
//template class GpuArray<int>;
//template class GpuArray<bool>;
