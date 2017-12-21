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

template <typename T>
__host__ __device__ __inline__ T Qe()       { return 1.602176462e-19; }
// velocity of light
template <typename T>
__host__ __device__ __inline__ T C()        { return 2.99792458e8; }        // m s^-1
// Planck's constant
template <typename T>
__host__ __device__ __inline__ T H()        { return 6.62606876e-34; }      // J s
// h-bar (h over 2 pi)
template <typename T>
__host__ __device__ __inline__ T Hbar()     { return 1.054571596e-34; }     // J s


template <typename T>
__host__ __device__ __inline__ T km2MeV(T km) {
   return km * 1E-3 * Qe<T>() / (Hbar<T>() * C<T>());
}


//TODO: avoid too many args
template <typename T>
__global__ void fullProb (int  start_id, 
			    T DMSq12, T DMSq13, T DMSq23,
			    T weight12, T weight13, T weight23, T weightCP,
				T km2, int EnuSize, T* devEnu, 
				T* devTmp, T* devComp0, T* devCompCP, 
				T* devComp12, T* devComp13, T* devComp23,
				T* ret, bool sameAB) {
  int x = (blockDim.x*blockIdx.x + threadIdx.x) ; // / streamcount + stream_id;
  if (x < 0 || x >= EnuSize) return;
  devTmp[x] = km2 / 2.0 * (1.0 / devEnu[x]);
  devComp0[x] = 1.0;
  devCompCP[x] = 0.0;
// TODO: add sharing
  T halfSin12, halfSin13, halfSin23,
         halfCos12, halfCos13, halfCos23;
  sincos(DMSq12 * devTmp[x] / 2.0, &halfSin12, &halfCos12);
  sincos(DMSq13 * devTmp[x] / 2.0, &halfSin13, &halfCos13);
  sincos(DMSq23 * devTmp[x] / 2.0, &halfSin23, &halfCos23);
 
// TODO: proove it really faster then sin and cos separately
  devComp12[x] = 2 * halfCos12 * halfCos12 - 1; 
  devComp13[x] = 2 * halfCos13 * halfCos13 - 1;
  devComp23[x] = 2 * halfCos23 * halfCos23 - 1;

  devCompCP[x] = ((T)!sameAB) * halfSin12 * halfSin13 * halfSin23;
  ret[x] = 2.0 * ( weight12 * devComp12[x]
		 + weight13 * devComp13[x]
		 + weight23 * devComp23[x]); 
// TODO: sharing
  T coeff0 = -2.0 * (weight12 + weight13 + weight23);
  coeff0 += (T)sameAB;
  ret[x] += coeff0 * devComp0[x];
  ret[x] += ((T)!sameAB) * 8.0 * weightCP * devCompCP[x];
}


template<typename T>
void calcCuFullProb(GNAcuOscProbMem<T> &mem,
                        T DMSq12, T DMSq13, T DMSq23,
			T weight12, T weight13, T weight23, T weightCP, 
			T* ret, T L, T* Enu, int EnuSize, bool sameAB) {
  const int blockSize = 16;
  int alloc_size = EnuSize * sizeof(T);
  cudaSetDevice(0);
 // GNAcuOscProbMem<double> mem(EnuSize);

  cudaError_t err;
std::cout << "EnuSize is " << EnuSize << std::endl;
  cudaStream_t stream1;
  cudaStreamCreate ( &stream1);

  bool tmp = false;

  T km2 = km2MeV<T>(L);

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
    if(i != streamcount - 1) cpySize = EnuSize*sizeof(T);
    else {
	cpySize = (fullEnuSize - i*EnuSize)*sizeof(T);
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
    fullProb<T><<<dataSize/blockSize + 1, blockSize, 0, workerstreams[i]>>>( i, DMSq12, DMSq13, DMSq23,
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

void calcCuFullProb_double (GNAcuOscProbMem<double> &mem,
                        double DMSq12, double DMSq13, double DMSq23,
                        double weight12, double weight13, double weight23, double weightCP,
                        double* ret, double L, double* Enu, int EnuSize, bool sameAB){
  calcCuFullProb<double>(mem, DMSq12, DMSq13, DMSq23,  weight12, weight13, weight23, weightCP, ret, L, Enu, EnuSize, sameAB);
}

void calcCuFullProb_float (GNAcuOscProbMem<float> &mem,
                                float DMSq12, float DMSq13, float DMSq23,
                                float weight12, float weight13, float weight23, float weightCP,
                                float* ret, float L, float* Enu, int EnuSize, bool sameAB) {
  calcCuFullProb<float>(mem, DMSq12, DMSq13, DMSq23,  weight12, weight13, weight23, weightCP, ret, L, Enu, EnuSize, sameAB);

}


