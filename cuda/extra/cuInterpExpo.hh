#pragma once

//extern "C" void interpExpo_v1(double** newx, double** newy, double* x, double* y,
//			 int** xsegments, double* xwidths, int oldsize, int newsize);

extern "C" void interpExpo_v1(double** args, double** rets, int Nnew, int Nold);

extern "C" void interpExpo_v2(double** newx, double** newy, double* x, double* y,
			 int** xsegments, double* xwidths, int oldsize, int newsize);

