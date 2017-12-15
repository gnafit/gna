#ifndef GNACUOSCPROBMEM_H
#define GNACUOSCPROBMEM_H

#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>


template<typename T>
class GNAcuOscProbMem {
// TODO: streams make here????
public:

  T* devEnu; 
  T* devTmp; 
  T* devComp0;
  T* devComp12; 
  T* devComp13; 
  T* devComp23;
  T* devCompCP; 
  T* devRet;

  GNAcuOscProbMem(int numOfElem) {
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

  ~GNAcuOscProbMem() {
    cudaFree(devComp0);
    cudaFree(devCompCP);
    cudaFree(devComp12);
    cudaFree(devComp13);
    cudaFree(devComp23);
    cudaFree(devRet);
    cudaFree(devTmp);
    cudaFree(devEnu);
  }
};

#endif /* GNACUOSCPROBMEM_H */

