#include <iostream>
#include "GNAcuOscProbFull.h"
#include <cstdlib>
/*
void calcCuFullProb (double DMSq12, double DMSq13, double DMSq23,
                                double weight12, double weight13, double weight23, double weightCP,
                                double* ret, double L, double* Enu, int EnuSize, bool sameAB);
*/



int main () {

  double DMSq12 = 0.5, DMSq13 = 0.4, DMSq23 = 0.2, weight12 = 1.0, weight13 = 1.0, weight23 = 1.0, weightCP = 1.0, L = 2;
  int EnuSize = 10;
  bool sameAB = false;
  double* ret = (double*)malloc(EnuSize*sizeof(double));
  double* Enu = (double*)malloc(EnuSize*sizeof(double));
  calcCuFullProb( DMSq12, DMSq13,  DMSq23,
                                 weight12,  weight13,  weight23,  weightCP,
                                 ret,  L, Enu, EnuSize,  sameAB);
  return 0;
}

