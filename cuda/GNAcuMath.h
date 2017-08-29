/*************************************************
Interfaces of CUDA implemetation of math methods
**************************************************/
#ifndef GNACUMATH_H
#define GNACUMATH_H

#include <cstdio>
#include <math.h>

extern "C" void cuInverseMat(int matSize, double* InMat, double* OutMat);
extern "C" void cuMultiplyMat(int m, int n, int k, double* InA, double* InB, double* OutC);
extern "C" void cuSolveLowerLS(int m, int n, double* A, double* B);

#endif /* GNACUMATH_H */
