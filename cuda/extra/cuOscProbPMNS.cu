#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <typeinfo>

#include "cuRootMath.h"
#include "../operations/cuOperations.cuh" 
#include <cuda.h>
#include "cuda_config_vars.h"
#include "cuOscProbPMNS.hh"


// TODO del debug values

//#define oscprobArgumentFactor 1.0
//#define m_L 1.0


// TODO define inern as array of the same as xarg size from internals


/*
 * CUDA version of calcComponent function in OscProbPMNS::OscProbPMNS (floats)
 * 
 */

/*
template <int I, int J>
__global__ void cuCalcComponent(float** xarg, float** xret, float** intern) {
	inverse(xarg, intern);
	prodNumToVec (cosf(DeltaMSq<I,J>() * oscprobArgumentFactor * m_L * 0.5f), 
				intern,
				xret);	
}
*/

/*
 * CUDA version of calcComponent function in OscProbPMNS::OscProbPMNS (doubles)
 * 
 */

//template < int I, int J>
__global__ void d_cuCalcComponent(double** xarg, double** xret, double** intern, unsigned int m) { 
	inverse(xarg[0], intern[0], m);
/*	prodNumToVec (cos( DeltaMSq<I,J>() * oscprobArgumentFactor * m_L * 0.5), 
				intern[0],
				xret[0]);	

*/
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < m) xret[0][idx] = intern[0][idx];
//	if (idx < m) {
//		intern[0][idx] = 10.0;
		xret[0][idx] = intern[0][idx];
//		xret[0][idx] = 10.0;
//	}
}


void cuCalcComponent(double** xarg, double** xret, double** intern, unsigned int m, unsigned int n) {
	d_cuCalcComponent<<<m/CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(xarg, xret, intern, m);
//	d_cuCalcComponent<<<1, n >>>(xarg, xret, intern, );
	cudaDeviceSynchronize();
}

/*
 * CUDA version of calcComponentCP function in OscProbPMNS::OscProbPMNS
 * 
 */

/*
template <typename T>
__global__ void cuCalcComponentCP(T** xarg, T** xret, T** intern) {
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
*/

