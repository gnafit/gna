#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <typeinfo>

#include "cuRootMath.h"
#include "../operations/cuOperations.cuh" 
#include <cuda.h>


// TODO del debug values

//#define oscprobArgumentFactor 1.0
//#define m_L 1.0


// TODO define inern as array of the same as xarg size from internals


/*
 * CUDA version of calcComponent function in OscProbPMNS::OscProbPMNS (floats)
 * 
 */

template <int I, int J>
__global__ void cuCalcComponent 
		(float* xarg, float* xret, float* intern) { 
	inverse(xarg, intern);
	prodNumToVec (cosf(DeltaMSq<I,J>() * oscprobArgumentFactor * m_L * 0.5f), 
				intern,
				xret);	
}


/*
 * CUDA version of calcComponent function in OscProbPMNS::OscProbPMNS (doubles)
 * 
 */

template < int I, int J>
__global__ void cuCalcComponent(double* xarg, double* xret, double* intern) { 
	inverse(xarg, intern);
	prodNumToVec (cos( DeltaMSq<I,J>() * oscprobArgumentFactor * m_L * 0.5), 
				intern,
				xret);	
}


/*
 * CUDA version of calcComponentCP function in OscProbPMNS::OscProbPMNS
 * 
 */

template <typename T>
__global__ void cuCalcComponentCP(T* xarg, T* xret, T* intern) {
	inverse(xarg, intern);
	prodNumToVec ((oscprobArgumentFactor * m_L * T(0.5)), intern, intern);
	arr_sin(DeltaMSq<1,2>(), intern, xret);
	mult_by_arr_sin(DeltaMSq<1,3>(), intern, xret);
	mult_by_arr_sin(DeltaMSq<2,3>(), intern, xret);
// for debug
//	arr_sin(1.0, intern, xret);
//	mult_by_arr_sin(1.0, intern, xret);
//	mult_by_arr_sin(1.0, intern, xret);
}

