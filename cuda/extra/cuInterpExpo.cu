#include <iostream>


__global__ void interp(double* newx, double* newy, double* x, double* y, 
			int* xsegments, double* xwidths) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int cur_xsegm = xsegments[idx];
	newy[idx] = y[cur_xsegm] * exp( -(newx[idx] - x[cur_xsegm]) *
					log(y[cur_xsegm] / y[cur_xsegm+1]) /
					xwidths[cur_xsegm]
					);
	// NOTE: if we need to spend a lot of time for xwidths transfer
	//       we can avoid it by computing it as
	// 	 x[cur_xsegm+1] - x[cur_xsegm];
}

