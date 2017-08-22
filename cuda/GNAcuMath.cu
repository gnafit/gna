#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include "cublas.h"
#include <typeinfo>
#include <iostream>
#include <cuda.h>


void cuInverseMat(int matSize, double* InMat, double* OutMat) {
  //if (typeid(InMat) != typeid(OutMat)) printf("Imput and output matrices types mismatch! Result can be wrong!"); 
/*  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  if (!InMat) {
     printf ("Input matrix is empty! Nothing to invert!");
     return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&devInMat, matSize*matSize*sizeof(*InMat));
  if (cudaStat != cudaSuccess) {
    printf ("Device memory allocation failed with input matrix");
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&devOutMat, matSize*matSize*sizeof(*OutMat));
  if (cudaStat != cudaSuccess) {
    printf ("Device memory allocation failed with output matrix");
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix(matSize, matSize, sizeof(InMat), InMat, M, devInMat, M); 
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("Input data download failed");
    cudaFree (devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }*/
  cublasHandle_t handle;
  cublasStatus_t ret;
  cudaError_t err;
  ret = cublasCreate(&handle);
  if(ret!=CUBLAS_STATUS_SUCCESS){
    printf("error code %d, line(%d)\n", ret, __LINE__);

switch (ret)
    {
        case CUBLAS_STATUS_SUCCESS:
            printf("CUBLAS_STATUS_SUCCESS "); exit(EXIT_FAILURE);

        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("CUBLAS_STATUS_NOT_INITIALIZED" );exit(EXIT_FAILURE);

        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("CUBLAS_STATUS_ALLOC_FAILED ");exit(EXIT_FAILURE);

        case CUBLAS_STATUS_INVALID_VALUE:
            printf("CUBLAS_STATUS_INVALID_VALUE ");exit(EXIT_FAILURE);

        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("CUBLAS_STATUS_ARCH_MISMATCH ");exit(EXIT_FAILURE);

        case CUBLAS_STATUS_MAPPING_ERROR:
            printf("CUBLAS_STATUS_MAPPING_ERROR ");exit(EXIT_FAILURE);

        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("CUBLAS_STATUS_EXECUTION_FAILED ");exit(EXIT_FAILURE);

        case CUBLAS_STATUS_INTERNAL_ERROR:
            printf("CUBLAS_STATUS_INTERNAL_ERROR ");exit(EXIT_FAILURE);
    }


    exit(EXIT_FAILURE);
  }
  //UINT wTimerRes = 0;
 // bool init = InitMMTimer(wTimerRes);
 // startTime = timeGetTime();
  double* devInMat;
  double* devOutMat;
  cudaMalloc((void**)&devInMat,  matSize*matSize*sizeof(*InMat));
  cudaMalloc((void**)&devOutMat,  matSize*matSize*sizeof(*OutMat));
  err = cudaMemcpy(devInMat, InMat, matSize*matSize*sizeof(*InMat), cudaMemcpyHostToDevice);
  if(err!=cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
  }
  //err = cudaMemcpy(D_B,B,nCols*nCols*sizeof(float),_HTD);
  if(err!=cudaSuccess) {
    printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
  }
//!!!!!!!!!
  const double alpha = 1.0;
  ret = cublasDtrsm_v2(handle,
          CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
          matSize, matSize, &alpha, devInMat, matSize, devOutMat, matSize);

  if(ret!=CUBLAS_STATUS_SUCCESS) {
    printf("error code %d, line(%d)\n", ret, __LINE__);
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(OutMat, devOutMat, matSize*matSize*sizeof(*OutMat), cudaMemcpyDeviceToHost);
  printf("InMat:\n");

  for (int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      printf("%lf ", InMat[i + matSize*j]);
      
    }
    printf("\n");
  }
  printf("OutMat:\n");
  for (int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      printf("%lf ", OutMat[i + matSize*j]);
    }
    printf("\n");
  }
  if(err!=cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
  }

  cudaFree(devInMat);
  cudaFree(devOutMat);

  //endTime = timeGetTime();
 // gtime=endTime-startTime;
  //std::cout << "GPU timing(including all device-host & host-device copies): " << float(gtime)/1000.0f << " seconds.\n";
  //DestroyMMTimer(wTimerRes, init);
}

int main() {
  double *InMat = new double[3*3];
  double *OutMat = new double[3*3];
  for(int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (j <= 3/2 && i >= 3/2) InMat[i+3*j] = 0.0; 
      else InMat[i+3*j] = i+j+1; 
    }
  }
  cuInverseMat(3, InMat, OutMat);
  //return 0;
}

/*int cuInverseMatInPlace(int matSize, double* InvMat) {

}*/
