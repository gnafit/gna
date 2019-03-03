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

__global__ void d_cuCalcComponent(double** xarg, double** xret, double** intern, double** params,
				unsigned int m, double oscprobArgumentFactor, double DeltaMSq, double m_L) { 
	inverse(xarg[0], intern[0], m);
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
        xret[0][idx] = cos( DeltaMSq * m_L * oscprobArgumentFactor * 0.5 * intern[0][idx]);      
}


void cuCalcComponent(double** xarg, double** xret, double** intern, double** params,
		 unsigned int m, unsigned int n, double oscprobArgumentFactor, double DeltaMSq, double m_L) {
	d_cuCalcComponent<<<m/CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(xarg, xret, intern, params,
								 m, oscprobArgumentFactor, DeltaMSq, m_L);
	cudaDeviceSynchronize();
}

/*
 * CUDA version of calcComponentCP function in OscProbPMNS::OscProbPMNS
 * 
 */


//template <typename T>
__global__ void d_cuCalcComponentCP(double** xarg, double** xret, double** intern, double** params, unsigned int m,
					double oscprobArgumentFactor, double m_L) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	inverse(xarg[0], intern[0], m);
	intern[0][idx] = oscprobArgumentFactor * m_L * 0.25 * intern[0][idx];
	arr_sin(params[0][0], intern[0], xret[0]);
	mult_by_arr_sin(params[0][1], intern[0], xret[0]);
	mult_by_arr_sin(params[0][2], intern[0], xret[0]);
// for debug
//	arr_sin(1.0, intern, xret);
//	mult_by_arr_sin(1.0, intern, xret);
//	mult_by_arr_sin(1.0, intern, xret);
}

void cuCalcComponentCP(double** xarg, double** xret, double** intern, double** params, 
			unsigned int m, unsigned int n, double oscprobArgumentFactor, double m_L) {
	d_cuCalcComponentCP<<<m/CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(xarg, xret, intern, params, m, 
								oscprobArgumentFactor, m_L);
	cudaDeviceSynchronize();
}
