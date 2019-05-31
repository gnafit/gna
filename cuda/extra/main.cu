//#include "cuOperations.cuh"

//#include "GNAcuOscProbFull.cu"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
//#include "cuOscProbPMNS.cu"

#include "cuInterpExpo.cu"

#define TYP double



int main() {
	cudaError_t err;

	int N = 100, M=100;
	TYP *host_array = new TYP[N];
        std::cout<<"INPUT IS "<<N<<" VECTORS:"<<std::endl;
/*        for (int i = 0; i < M; i++){
                host_array[i] = TYP(i);
                std::cout<<host_array[i]<<" ";
        }
*/
        std::cout<<std::endl;
        std::cout<<std::endl;

	
	TYP *dev_ptr_array;
	TYP *dev_ans;
	TYP *dev_tmp;

	err = cudaMalloc(&dev_ptr_array, N * sizeof(TYP*));
	err = cudaMalloc(&dev_ans, N * sizeof(TYP*));
	err = cudaMalloc(&dev_tmp, N * sizeof(TYP*));
	err = cudaMemcpy(dev_ptr_array, host_array, N * sizeof(TYP*), cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_tmp, host_array, N * sizeof(TYP*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}


	int Nold = 10, Nnew = 20;
	double *h_newx = new double[Nnew];
	double *h_newx2 = new double[Nnew];
	double *h_newy = new double[Nnew];
	double *h_newy2 = new double[Nnew];
	double *h_x = new double[Nold];
	double *h_y = new double[Nold];
	double *h_xwidth = new double[Nold];
	int *h_xsegments = new int[Nnew];
	int *h_xsegments2 = new int[Nnew];
	
	double *d_newx ;
	double *d_newx2 ;
	double *d_newy ;
	double *d_newy2 ;
	double *d_x ;
	double *d_y ;
	double *d_xwidth ;
	int *d_xsegment;
	int *d_xsegment2;


	int K = 2;
	double **h_newx_uni = (double**)malloc(K * sizeof(double*));
	double **h_newy_uni = (double**)malloc(K * sizeof(double*));
	int** h_xsegments_uni = (int**)malloc(K * sizeof(int*));

	double **d_newx_uni;
        cudaMalloc(&d_newx_uni, K * sizeof(double*));
	double **d_newy_uni;
        cudaMalloc(&d_newy_uni, K * sizeof(double*));
	int** d_xsegments_uni;
        cudaMalloc(&d_xsegments_uni, K * sizeof(double*));

	cudaMalloc(&d_newx, Nnew * sizeof(double));
	cudaMalloc(&d_newx2, Nnew * sizeof(double));
	cudaMalloc(&d_newy, Nnew * sizeof(double));
	cudaMalloc(&d_newy2, Nnew * sizeof(double));
	cudaMalloc(&d_x, Nold * sizeof(double));
	cudaMalloc(&d_y, Nold * sizeof(double));
	cudaMalloc(&d_xwidth, Nold * sizeof(double));
	cudaMalloc(&d_xsegment, Nnew * sizeof(double));
	cudaMalloc(&d_xsegment2, Nnew * sizeof(double));


	for (int i = 0; i < Nnew; i++) {
//		h_newy[i] = i/2.0 + 1;
		h_newx[i] = i/4.0 + 1;
		h_newx2[i] = i/4.0 + 6;
		h_xsegments[i] = int(i/4);
		h_xsegments2[i] = int(i/4)+5;
	}
	for(int i = 0; i < Nold; i++) {
		h_x[i] = i +1;
		h_y[i] = i +1;
		h_xwidth[i] = 1;
	}
	
        err = cudaMemcpy(d_newx, h_newx, Nnew * sizeof(double), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_newx2, h_newx2, Nnew * sizeof(double), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_xwidth, h_xwidth, Nold * sizeof(double), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_x, h_x, Nold * sizeof(double), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_y, h_y, Nold * sizeof(double), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_xsegment, h_xsegments, Nnew * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_xsegment2, h_xsegments2, Nnew * sizeof(int), cudaMemcpyHostToDevice);

	h_newx_uni[0] = d_newx;
	h_newx_uni[1] = d_newx2;
	h_newy_uni[0] = d_newy;
	h_newy_uni[1] = d_newy2;

	h_xsegments_uni[0] = d_xsegment;
	h_xsegments_uni[1] = d_xsegment2;

	cudaMemcpy( d_newx_uni, h_newx_uni, K*sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy( d_newy_uni, h_newy_uni, K*sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy( d_xsegments_uni, h_xsegments_uni, K*sizeof(double*), cudaMemcpyHostToDevice);


	std::cout << "segments: " << std::endl;
	for (int i = 0; i <Nnew; i++) {
		//std::cout << h_xsegments[i] << " ";
		std::cout << h_newx[i] << " ";
	}
	for (int i = 0; i <Nnew; i++) {
		//std::cout << h_xsegments2[i] << " ";
		std::cout << h_newx2[i] << " ";
	}
	std::cout << std::endl; 
	
	
	
	interpExpo<<<1, dim3(K,Nnew)>>> ( d_newx_uni, d_newy_uni,  d_x,  d_y,d_xsegments_uni, d_xwidth, Nold);
	//tmpf<<<1, dim3(K,Nnew)>>> ( d_newx_uni, d_newy_uni,  d_x,  d_y,d_xsegments_uni, d_xwidth, Nold);
//	cuCalcComponentCP<<<1,N>>>(dev_ptr_array, dev_ans, dev_tmp);
	cudaDeviceSynchronize();

	cudaMemcpy(h_newy, d_newy, Nnew*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_newy2, d_newy2, Nnew*sizeof(double), cudaMemcpyDeviceToHost);
	
	std::cout << "segments: " << std::endl;
        for (int i = 0; i <Nnew; i++) {
                std::cout << h_newy[i] << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i <Nnew; i++) {
                std::cout << h_newy2[i] << " ";
        }
        std::cout << std::endl;
	
	//err = cudaMemcpy( host_array, dev_ans, N * sizeof(TYP*), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}


/*        for (int i = 0; i < M; i++){
                std::cout<<host_array[i]<<" ";
        }
  */      std::cout<<std::endl;
	return 0;
}
