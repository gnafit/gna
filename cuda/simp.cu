#include <cstdio>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simp.h"
#include "cublas_v2.h"
#define M 6
#define N 5
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
	    cublasSscal (handle, n-p+1, &alpha, &m[IDX2F(p,q,ldm)], ldm);
	        cublasSscal (handle, ldm-p+1, &beta, &m[IDX2F(p,q,ldm)], 1);
}


int notmain (void){
	cudaError_t cudaStat;    
	cublasStatus_t stat;
	cublasHandle_t handle;
	int i, j;
	float* devPtrA;
	float* a = 0;
	a = (float *)malloc (M * N * sizeof (*a));
	if (!a) {
		printf ("host memory allocation failed");
		return EXIT_FAILURE;
	}
	for (j = 1; j <= N; j++) {
		for (i = 1; i <= M; i++) {
			a[IDX2F(i,j,M)] = (float)((i-1) * M + j);
		}
	}
	cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
	if (cudaStat != cudaSuccess) {
		printf ("device memory allocation failed");
		return EXIT_FAILURE;
	}
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
		return EXIT_FAILURE;
	}
	stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("data download failed");
		cudaFree (devPtrA);
		cublasDestroy(handle);
		return EXIT_FAILURE;
	}
	modify (handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
	stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("data upload failed");
		cudaFree (devPtrA);
		cublasDestroy(handle);        
		return EXIT_FAILURE;
	}    
	cudaFree (devPtrA);
	cublasDestroy(handle);
	for (j = 1; j <= N; j++) {
		for (i = 1; i <= M; i++) {
			printf ("%7.0f", a[IDX2F(i,j,M)]);
		}
		printf ("\n");
	}
	free(a);
	return EXIT_SUCCESS;
}

const int NN = 50;

__global__ void  add( int *a,int *b, int *c ) {
int tid = blockIdx.x;    
if (tid < NN)
        c[tid] = a[tid] + b[tid];
}

void simpfun() {
    int  a[NN], b[NN], c[NN];
    int  *dev_a, *dev_b, *dev_c;
    cudaSetDevice(0);
    cudaMalloc( (void**)&dev_a, NN * sizeof(int ) );
    cudaMalloc( (void**)&dev_b, NN * sizeof(int) );
    cudaMalloc( (void**)&dev_c, NN * sizeof(int) );
    
    for (int i=0; i<NN; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy( dev_a, a, NN * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, NN * sizeof(int), cudaMemcpyHostToDevice );
    
    add<<<NN,1>>>( dev_a, dev_b, dev_c ); 
    
    cudaMemcpy( c, dev_c, NN * sizeof(int), cudaMemcpyDeviceToHost );
    for (int i=0; i<NN; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i] );
    }
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    int kkk = notmain();
    std::cout << "KKK is here: " << kkk << std::endl;
}
