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
template<typename T>
__global__ void d_cuCalcComponent(T** xarg, T** xret, T** intern, T** params,
				unsigned int m, T oscprobArgumentFactor, T DeltaMSq, T m_L) { 
	inverse(xarg[0], intern[0], m);
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
        xret[0][idx] = cos( DeltaMSq * m_L * oscprobArgumentFactor * 0.5 * intern[0][idx]);      
}


template<typename T>
void cuCalcComponent(T** xarg, T** xret, T** intern, T** params,
		 unsigned int m, unsigned int n, T oscprobArgumentFactor, T DeltaMSq, T m_L) {
	d_cuCalcComponent<<<m/CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(xarg, xret, intern, params,
								 m, oscprobArgumentFactor, DeltaMSq, m_L);
	cudaDeviceSynchronize();
}

/*
 * CUDA version of calcComponentCP function in OscProbPMNS::OscProbPMNS
 * 
 */
template <typename T>
__global__ void d_cuCalcComponentCP(T** xarg, T** xret, T** intern, T** params, T m12, T m13, T m23, T int m,
					T oscprobArgumentFactor, T m_L) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	xret[0][idx] = 1.0;
	inverse(xarg[0], intern[0], m);
	intern[0][idx] = oscprobArgumentFactor * m_L * 0.25 * intern[0][idx];
	arr_sin(m12, intern[0], xret[0]);
//	arr_sin(1.0, intern[0], xret[0]);
	
	mult_by_arr_sin(m13, intern[0], xret[0]);
	mult_by_arr_sin(m23, intern[0], xret[0]);

// for debug
//	arr_sin(1.0, intern, xret);
//	mult_by_arr_sin(1.0, intern, xret);
//	mult_by_arr_sin(1.0, intern, xret);
}

template<typename T>
void cuCalcComponentCP(T** xarg, T** xret, T** intern, T** params, T m12, T m13, T m23, 
			unsigned int m, unsigned int n, T oscprobArgumentFactor, T m_L) {
	d_cuCalcComponentCP<<<m/CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(xarg, xret, intern, params, m12, m13, m23, m, 
								oscprobArgumentFactor, m_L);
	cudaDeviceSynchronize();
}

template<typename T>
__global__ void d_cuCalcSum(T** xarg, T** xret, T w12, T w13, T w23, T wcp, bool isSame) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
/*	xret[0][idx]=2.0*(params[0][0]*xarg[0][idx]+params[0][1]*xarg[1][idx]+params[0][2]*xarg[2][idx])
			+ (-2.0*(params[0][0]+params[0][1]+params[0][2]) 
			+ 1.0*isSame) * xarg[3][idx]
			+ 8.0*(!isSame)*params[0][4]*xarg[4][idx];
*/
	xret[0][idx]=2.0*(w12*xarg[0][idx]+w13*xarg[1][idx]+w23*xarg[2][idx])
			+ (-2.0*(w12+w13+w23) 
			+ 1.0*isSame) * xarg[3][idx]
			+ 8.0*(!isSame)*wcp*xarg[4][idx];
}

template<typename T>
void cuCalcSum(T** xarg, T** xret, T w12, T w13, T w23, T wcp, bool isSame, unsigned int m) {
	d_cuCalcSum<<<m/CU_BLOCK_SIZE + 1, CU_BLOCK_SIZE>>>(xarg, xret, w12, w13, w23, wcp, isSame);
	cudaDeviceSynchronize();
}

template void cuCalcComponent<double>(double** xarg, double xret,  double w12, double w13, double w23, double wcp, bool isSame, unsigned int m);
template void cuCalcComponent<float>(float** xarg, float xret,  float w12, float w13, float w23, float wcp, bool isSame, unsigned int m);

template void cuCalcComponentCP<double>(double** xarg, double** xret, double** intern, double** params, double m12, double m13, double m23, unsigned int m, unsigned int n, double oscprobArgumentFactor, double m_L);
template void cuCalcComponentCP<float>(float** xarg, float** xret, float** intern, float** params, float m12, float m13, float m23, unsigned int m, unsigned int n, float oscprobArgumentFactor, float m_L);

template void cuCalcComponent<double>(double** xarg, double** xret, double** intern, double** params, unsigned int m, unsigned int n, double oscprobArgumentFactor, double DeltaMSq, double m_L);
template void cuCalcComponent<float>(float** xarg, float** xret, float** intern, float** params, unsigned int m, unsigned int n, float oscprobArgumentFactor, float DeltaMSq, float m_L);

