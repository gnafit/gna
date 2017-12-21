#include "GNAcuOscProbMem.hh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template<typename T>
GNAcuOscProbMem<T>::GNAcuOscProbMem(int numOfElem) {
    cudaError_t err;
    size_t alloc_size = sizeof(T) * numOfElem;
    cudaSetDevice(0);

    err = cudaMalloc((void**)&devEnu, alloc_size);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to  allocate!\n");
      std::cout << "err is " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc((void**)&devTmp, alloc_size);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to  allocate!\n");
    }
    err = cudaMalloc((void**)&devComp0, alloc_size);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to  allocate!\n");
    }
    err = cudaMalloc((void**)&devComp12, alloc_size);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to  allocate!\n");
    }
    err = cudaMalloc((void**)&devComp13, alloc_size);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to  allocate!\n");
    }
    err = cudaMalloc((void**)&devComp23, alloc_size);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to  allocate!\n");
    }
    err = cudaMalloc((void**)&devCompCP, alloc_size);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to  allocate!\n");
    }
    err = cudaMalloc((void**)&devRet, alloc_size);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to  allocate!\n");
    }
}

template<typename T>
GNAcuOscProbMem<T>::~GNAcuOscProbMem() {
    cudaFree(devComp0);
    cudaFree(devCompCP);
    cudaFree(devComp12);
    cudaFree(devComp13);
    cudaFree(devComp23);
    cudaFree(devRet);
    cudaFree(devTmp);
    cudaFree(devEnu);
}

template class GNAcuOscProbMem<double>;
template class GNAcuOscProbMem<float>;

