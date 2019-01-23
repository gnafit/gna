#include <iostream>



/* Exp interpolation with segment indexes started with -1 in old way. 
 *
 */
__global__ void interpExpo(double* newx, double* newy, double* x, double* y, 
			int* xsegments, double* xwidths, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int idy = blockIdx.y * blockDim.y + threadIdx.y;

	//int cur_xsegm = xsegments[idy][idx];
	int cur_xsegm = xsegments[idx];

	// check for underflow and overflow
	if (cur_xsegm < 0) {
		cur_xsegm = 0;
	}
	if (cur_xsegm >= n-1) { 
		cur_xsegm = n-2; 
	} 

	//newy[idy][idx] = y[cur_xsegm] * exp( -(newx[idy][idx] - x[cur_xsegm]) *
	newy[idx] = y[cur_xsegm] * exp( -(newx[idx] - x[cur_xsegm]) *
					log(y[cur_xsegm] / y[cur_xsegm+1]) /
					xwidths[cur_xsegm]
					);

	// NOTE: if we need to spend a lot of time for xwidths transfer
	//       we can avoid it by computing it as
	// 	 x[cur_xsegm+1] - x[cur_xsegm];
}

