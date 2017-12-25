#include <iostream>
#include "GNAcuOscProbFull.h"
#include <cstdlib>
#include "GNAcuRootMath.cuh"
#include "GNAcuOscProbMem.hh"

/*
void calcCuFullProb (double DMSq12, double DMSq13, double DMSq23,
                                double weight12, double weight13, double weight23, double weightCP,
                                double* ret, double L, double* Enu, int EnuSize, bool sameAB);
*/



int main () {

  int EnuSize = 10000000;
  bool sameAB = false;

  double DMSq12 = 0.5, DMSq13 = 0.4, DMSq23 = 0.2, weight12 = 1.0, weight13 = 1.0, weight23 = 1.0, weightCP = 1.0, L = 2;
  GNAcuOscProbMem<double> mem(EnuSize);
  double* ret = (double*)malloc(EnuSize*sizeof(double));
  double* Enu = (double*)malloc(EnuSize*sizeof(double));
  calcCuFullProb_double(mem, DMSq12, DMSq13,  DMSq23,
                                 weight12,  weight13,  weight23,  weightCP,
                                 ret,  L, Enu, EnuSize,  sameAB);
/*
  float DMSq12f = 0.5, DMSq13f = 0.4, DMSq23f = 0.2, weight12f = 1.0, weight13f = 1.0, weight23f = 1.0, weightCPf = 1.0, Lf = 2;
  GNAcuOscProbMem<float> memf(EnuSize);
  float* retf = (float*)malloc(EnuSize*sizeof(float));
  float* Enuf = (float*)malloc(EnuSize*sizeof(float));
  calcCuFullProb_float(memf, DMSq12f, DMSq13f,  DMSq23f,
                                 weight12f,  weight13f,  weight23f,  weightCPf,
                                 retf,  Lf, Enuf, EnuSize,  sameAB);
*/
  return 0;
}

