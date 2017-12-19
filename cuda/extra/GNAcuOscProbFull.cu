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
#include <iostream>
#include "GNAcuOscProbMem.hh"

__host__ __device__ __forceinline__ double Qe()       { return 1.602176462e-19; }
// velocity of light
__host__ __device__ __forceinline__ double C()        { return 2.99792458e8; }        // m s^-1
// Planck's constant
__host__ __device__ __forceinline__ double H()        { return 6.62606876e-34; }      // J s
// h-bar (h over 2 pi)
__host__ __device__ __forceinline__ double Hbar()     { return 1.054571596e-34; }     // J s


__host__ __device__ __forceinline__ double km2MeV(double km) {
   return km * 1E-3 * Qe() / (Hbar() * C());
}


//TODO: avoid too many args
__global__ void fullProb (int  start_id, 
			    double DMSq12, double DMSq13, double DMSq23,
			    double weight12, double weight13, double weight23, double weightCP,
				double km2, int EnuSize, double* devEnu, 
				double* devTmp, double* devComp0, double* devCompCP, 
				double* devComp12, double* devComp13, double* devComp23,
				double* ret, bool sameAB) {
  int x = (blockDim.x*blockIdx.x + threadIdx.x) ; // / streamcount + stream_id;
  if (x < 0 || x >= EnuSize) return;
  devTmp[x] = km2 / 2.0 * (1.0 / devEnu[x]);
  devComp0[x] = 1.0;
  devCompCP[x] = 0.0;
// TODO: add sharing
  double halfSin12, halfSin13, halfSin23,
         halfCos12, halfCos13, halfCos23;
  sincos(DMSq12 * devTmp[x] / 2.0, &halfSin12, &halfCos12);
  sincos(DMSq13 * devTmp[x] / 2.0, &halfSin13, &halfCos13);
  sincos(DMSq23 * devTmp[x] / 2.0, &halfSin23, &halfCos23);
 
// TODO: proove it really faster then sin and cos separately
  devComp12[x] = 2 * halfCos12 * halfCos12 - 1; 
  devComp13[x] = 2 * halfCos13 * halfCos13 - 1;
  devComp23[x] = 2 * halfCos23 * halfCos23 - 1;

  devCompCP[x] = ((double)!sameAB) * halfSin12 * halfSin13 * halfSin23;
  ret[x] = 2.0 * ( weight12 * devComp12[x]
		 + weight13 * devComp13[x]
		 + weight23 * devComp23[x]); 
// TODO: sharing
  double coeff0 = -2.0 * (weight12 + weight13 + weight23);
  coeff0 += (double)sameAB;
  ret[x] += coeff0 * devComp0[x];
  ret[x] += ((double)!sameAB) * 8.0 * weightCP * devCompCP[x];
}

void calcCuFullProb(GNAcuOscProbMem<double> &mem,
                        double DMSq12, double DMSq13, double DMSq23,
			double weight12, double weight13, double weight23, double weightCP, 
			double* ret, double L, double* Enu, int EnuSize, bool sameAB) {
  const int blockSize = 16;
  int alloc_size = EnuSize * sizeof(double);
  cudaSetDevice(0);
 // GNAcuOscProbMem<double> mem(EnuSize);

  cudaError_t err;
std::cout << "EnuSize is " << EnuSize << std::endl;
  cudaStream_t stream1;
  cudaStreamCreate ( &stream1);

  bool tmp = false;
  double ttt = 5.5;
  std::cout << "test = " << ttt + tmp << " " << ttt*tmp <<  std::endl;

  double km2 = km2MeV(L);

  std::cout << "km2 = " << km2 << std::endl;

  int streamcount = 8;
  cudaStream_t workerstreams[streamcount];
  int fullEnuSize = EnuSize;
  EnuSize /= streamcount;

  cudaDeviceSynchronize();

  for (int i = 0; i < streamcount; i++) {
    int k = i*EnuSize;
    cudaStreamCreate ( &workerstreams[i]);
    int cpySize;
    if(i != streamcount - 1) cpySize = EnuSize*sizeof(double);
    else {
	cpySize = (fullEnuSize - i*EnuSize)*sizeof(double);
    }
    err = cudaMemcpyAsync((void**)&mem.devEnu[k], (void**)&Enu[k], cpySize, cudaMemcpyHostToDevice, workerstreams[i]);
    if(err!=cudaSuccess) {
      printf("ERROR: unable to copy memory from host to device in for! \n");
      std::cout << "err is " << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  for (int i = 0; i < streamcount; i++) {
    int k = i*EnuSize;
    int dataSize;
    if(i != streamcount - 1) dataSize = EnuSize;
    else {
        dataSize = fullEnuSize - (i-1)*EnuSize;
    }
    fullProb<<<dataSize/blockSize + 1, blockSize, 0, workerstreams[i]>>>( i, DMSq12, DMSq13, DMSq23,
                   weight12, weight13, weight23, weightCP,
                   km2, dataSize, &mem.devEnu[k],
                   &mem.devTmp[k], &mem.devComp0[k], &mem.devCompCP[k],
                   &mem.devComp12[k], &mem.devComp13[k], &mem.devComp23[k],
                   &mem.devRet[k], sameAB);
  }
  cudaDeviceSynchronize(); 
//  TODO: Where do we need to do sync?
  err = cudaMemcpyAsync(ret, mem.devRet, alloc_size, cudaMemcpyDeviceToHost, stream1);
  if(err!=cudaSuccess) {
    printf("ERROR: unable to copy memory from device to host! \n");
    std::cout << "err is " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  cudaStreamDestroy(stream1);
  for (int i = 0 ; i < streamcount; i++) {
    cudaStreamDestroy(workerstreams[i]);
  }
}

