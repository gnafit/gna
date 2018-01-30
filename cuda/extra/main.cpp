#include <iostream>
#include "GNAcuOscProbFull.h"
#include <cstdlib>
//#include "GNAcuRootMath.cuh"
#include "GNAcuOscProbMem.hh"
#include "GNAcuGpuArray.hh"
/*
void calcCuFullProb (double DMSq12, double DMSq13, double DMSq23,
                                double weight12, double weight13, double weight23, double weightCP,
                                double* ret, double L, double* Enu, int EnuSize, bool sameAB);
*/



int main () {

  int EnuSize = 100;
  bool sameAB = false;

  double* ret1 = new double[EnuSize];// = (double*)malloc(EnuSize*sizeof(double));
  for(int i = 0; i < EnuSize; i++) {
    ret1[i] = i;
  }
  double* ret2;// = (double*)malloc(EnuSize*sizeof(double));
  GNAcuGpuArray<double> arr1( EnuSize);
  GNAcuGpuArray<double> arr2( EnuSize);
  arr1.setByValue(5.0);
  arr2.setByValue(3.0);
  double* answ;// = (double*)malloc(EnuSize*sizeof(double));
  GNAcuGpuArray<double> arrAnsw (EnuSize);
  //arrAnsw = -arr2+arr1*10.0;
  arrAnsw.setByHostArray(ret1);
  std::cout << "size = " << arrAnsw.getArraySize();
  double* res = new double[EnuSize]; 
//  (arr1 + arr2).getContentToCPU(res);
  arrAnsw.getContentToCPU(res);
  for (int i = 0; i < EnuSize; i++) {
	std::cout << res[i] << " ";
  }

/*  double DMSq12 = 0.5, DMSq13 = 0.4, DMSq23 = 0.2, weight12 = 1.0, weight13 = 1.0, weight23 = 1.0, weightCP = 1.0, L = 2;
  GNAcuOscProbMem<double> mem(EnuSize);
  double* ret = (double*)malloc(EnuSize*sizeof(double));
  double* Enu = (double*)malloc(EnuSize*sizeof(double));
  calcCuFullProb_double(mem, DMSq12, DMSq13,  DMSq23,
                                 weight12,  weight13,  weight23,  weightCP,
                                 ret,  L, Enu, EnuSize,  sameAB);
*/
/*
  float DMSq12f = 0.5f, DMSq13f = 0.4f, DMSq23f = 0.2f, weight12f = 1.0f, weight13f = 1.0f, weight23f = 1.0f, weightCPf = 1.0f, Lf = 2.0f;
  GNAcuOscProbMem<float> memf(EnuSize);
  float* retf = (float*)malloc(EnuSize*sizeof(float));
  float* Enuf = (float*)malloc(EnuSize*sizeof(float));
  calcCuFullProb_float(memf, DMSq12f, DMSq13f,  DMSq23f,
                                 weight12f,  weight13f,  weight23f,  weightCPf,
                                 retf,  Lf, Enuf, EnuSize,  sameAB);
*/
  return 0;
}

