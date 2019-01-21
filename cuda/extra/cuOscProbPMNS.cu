#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <typeinfo>

#include "cuRootMath.h"
#include "cuMathFunctions.h"
#include "cuOperations.cuh" 
#include <cuda.h>

/* TMP */

#define m_L 1.1
#define oscprobArgumentFactor 1.1

template<>
double OscProbPMNSBase::DeltaMSq<1,2>() const {return 0.1; }

template<>
double OscProbPMNSBase::DeltaMSq<1,3>() const {return 0.1; }

template<>
double OscProbPMNSBase::DeltaMSq<2,3>() const {return 0.1; }


/* TMP END */




template <int I, int J, typename T>
__global__ void cuCalcComponent(T* xarg, T* xret, T* intern); 
// TODO define inern as array of the same as xarg size from internals


template <int I, int J>
__global__ void cuCalcComponent(float* xarg, float* xret, float* intern) { 
	inverse(xarg, intern);
	prodNumToVec (cosf(DeltaMSq<I,J>() * oscprobArgumentFactor * m_L * 0.5f), 
				intern,
				xret);	
}


template <int I, int J>
__global__ void cuCalcComponent(double* xarg, double* xret, double* intern) { 
	inverse(xarg, intern);
	prodNumToVec (cosf(DeltaMSq<I,J>() * oscprobArgumentFactor * m_L * 0.5), 
				intern,
				xret);	
}

