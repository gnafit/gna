#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <typeinfo>
#include <iostream>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "GNAcuOscProbFull.h"
#include "math_functions.h"

//#include "Constants.h"


__host__ __device__ __forceinline__  double Qe()       { return 1.602176462e-19; }
// velocity of light
__host__ __device__ __forceinline__  double C()        { return 2.99792458e8; }        // m s^-1
// Planck's constant
__host__ __device__ __forceinline__  double H()        { return 6.62606876e-34; }      // J s
// h-bar (h over 2 pi)
__host__ __device__ __forceinline__  double Hbar()     { return 1.054571596e-34; }     // J s


__host__ __device__ double km2MeV(double km) {
   return km * 1E-3 * Qe() / (Hbar() * C());
}

//TODO: avoid too mane args
__global__ void fullProb (double DMSq12, double DMSq13, double DMSq23,
			    double weight12, double weight13, double weight23, double weightCP,
				double km2, int EnuSize, double* devEnu, 
				double* devTmp, double* devComp0, double* devCompCP, 
				double* devComp12, double* devComp13, double* devComp23,
				double* ret, bool sameAB) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
//  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if (x < 0 || x >= EnuSize) return;
  devTmp[x] = km2 / 2.0 * (1.0 / devEnu[x]);
  devComp0[x] = 1.0;
  devCompCP[x] = 0.0;
// TODO: add sharing
  double halfSin12, halfSin13, halfSin23,
         halfCos12, halfCos13, halfCos23;
// TODO: add streams
  sincos(DMSq12 * devTmp[x] / 2.0, &halfSin12, &halfCos12);
  sincos(DMSq13 * devTmp[x] / 2.0, &halfSin13, &halfCos13);
  sincos(DMSq23 * devTmp[x] / 2.0, &halfSin23, &halfCos23);
 
// TODO: proove it really faster then sin and cos separately
  devComp12[x] = 2 * halfCos12 * halfCos12 - 1; 
  devComp13[x] = 2 * halfCos13 * halfCos13 - 1;
  devComp23[x] = 2 * halfCos23 * halfCos23 - 1;
// TODO: avoid ifs
  if (!sameAB) {
    devCompCP[x] = halfSin12 * halfSin13 * halfSin23;
  }
  ret[x] = 2.0 * ( weight12 * devComp12[x]
		 + weight13 * devComp13[x]
		 + weight23 * devComp23[x]); 
// TODO: sharing
  double coeff0 = -2.0 * (weight12 + weight13 + weight23);
// TODO: avoid ifs
  if (sameAB) {
    coeff0 += 1.0;
  }
  ret[x] += coeff0 * devComp0[x];
  if (!sameAB) {
    ret[x] += 8.0 * weightCP * devCompCP[x];
  }
}

void calcCuFullProb(double DMSq12, double DMSq13, double DMSq23,
			double weight12, double weight13, double weight23, double weightCP, 
			double* ret, double L, double* Enu, int EnuSize, bool sameAB) {
// TODO: avoid cublas

  const int blockSize = 16;
  int alloc_size = EnuSize * sizeof(double);
  cudaSetDevice(0);
  //cublasHandle_t handle;
  //cublasStatus_t ret;
  cudaError_t err;

  cudaStream_t stream1, stream2;
  cudaStreamCreate ( &stream1);
  cudaStreamCreate ( &stream2);
  
 /* ret = cublasCreate(&handle);
  if(ret!=CUBLAS_STATUS_SUCCESS){
    printf("ERROR: unable to create cuBLAS handle!\n");
    exit(EXIT_FAILURE);
  }
*/
  /* Allocating device memory */
  double* devEnu; double* devTmp; double* devComp0;
  double* devComp12; double* devComp13; double* devComp23;
  double* devCompCP; double* devRet;
  cudaMalloc((void**)&devEnu, alloc_size);

  err = cudaMemcpyAsync(devEnu, Enu, alloc_size, cudaMemcpyHostToDevice, stream1);
  cudaMalloc((void**)&devTmp, alloc_size);
  cudaMalloc((void**)&devComp0, alloc_size);
  cudaMalloc((void**)&devComp12, alloc_size);
  cudaMalloc((void**)&devComp13, alloc_size);
  cudaMalloc((void**)&devComp23, alloc_size);
  cudaMalloc((void**)&devCompCP, alloc_size);
  cudaMalloc((void**)&devRet, alloc_size);

  if(err!=cudaSuccess) {
    printf("ERROR: unable to copy memory from host to device! \n");
    exit(EXIT_FAILURE);
  }
  double km2 = km2MeV(L);
// TODO: choose call grid parameters
  fullProb<<<1, EnuSize>>>(DMSq12, DMSq13, DMSq23,
                   weight12, weight13, weight23, weightCP,
                   km2, EnuSize, devEnu,
                   devTmp, devComp0, devCompCP,
                   devComp12,  devComp13, devComp23,
                   ret, sameAB);
 
//  TODO: Where we need to do sync?
  err = cudaMemcpyAsync(ret, devRet, alloc_size, cudaMemcpyDeviceToHost, stream1);
  if(err!=cudaSuccess) {
    printf("ERROR: unable to copy memory from host to device! \n");
    exit(EXIT_FAILURE);
  }
  cudaFree(&devComp0);   cudaFree(&devCompCP);
  cudaFree(&devComp12);  cudaFree(&devComp13);  cudaFree(&devComp23);
  cudaFree(&devRet);     cudaFree(&devTmp);     cudaFree(&devEnu);
  
}

void test (double* data) {
// TODO: delete all Thrust traces 
  for (int i = 0; i < 10; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
  thrust::host_vector<double> DoubVec (data, data+10);
  for (int i = 0; i < 10; i++) {
    std::cout << DoubVec[i] << " ";
  }
}
