/*************************************************
Interfaces of CUDA implemetation of math methods
**************************************************/
#include <cstdio>
#include <math.h>

extern "C" void cuInverseMat(int matSize, double* InMat, double* OutMat);
