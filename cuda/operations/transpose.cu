#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm> 
#include <math.h>

const int CU_BLOCK_SIZE = 32;
#define GridSize(size) (size/CU_BLOCK_SIZE + 1)

__global__
void transpose(double** array, double** ans_array, int* n, int* m, int amount) {
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	if(Col < n[0] && Row < m[0]) { //filter unneeded threads
		int numCColumns = n[0];
		int numCRows = m[0];
		if ((Row < m[0]) && (Col < n[0]))
			ans_array[0][Col*numCRows + Row] = array[0][Row*numCColumns + Col]; //first matrix >> answer
		n[0] = numCRows;
		m[0] = numCColumns;
	}
}

void init(int N, int M, int K1, int K2){
	cudaError_t err;
	cudaSetDevice(0);

	double *host_array = new double[M];
	std::cout<<"INPUT IS "<<N<<" VECTORS:"<<std::endl;
	for (int i = 0; i < M; i++){
		host_array[i] = i;
		std::cout<<host_array[i]<<" ";
	}
	std::cout<<std::endl;

	double **host_ptr_array;

	host_ptr_array = (double**)malloc(N*sizeof(double*));

//GPU memory allocation	
	for (int i = 0; i < N; i++){
		cudaMalloc((void**)&host_ptr_array[i], M * sizeof(double));
		err =
			cudaMemcpy(host_ptr_array[i], host_array, M * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "1Err is " << cudaGetErrorString(err) << std::endl;
		}
	}

	double **dev_ptr_array;
	cudaMalloc(&dev_ptr_array, N * sizeof(double*));
	err =
		cudaMemcpy(dev_ptr_array, host_ptr_array, N * sizeof(double*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}

//Dimensions allocation
	int *dim1_array = new int[N];
	int *dev_dim1_array;
	int *dim2_array = new int[N];
	int *dev_dim2_array;
	std::cout<<"DIMS ARE:"<<std::endl;
	for (int i = 0; i < N; i++){
		if (i % 2 == 0){
			dim1_array[i] = K1;
			dim2_array[i] = K2;
		} else{
			dim1_array[i] = K2;
			dim2_array[i] = K1;
		}
		std::cout<<dim1_array[i]<<" "<<dim2_array[i]<<std::endl;;
	}
	std::cout<<std::endl;
	cudaMalloc(&dev_dim1_array, N * sizeof(int));
	cudaMalloc(&dev_dim2_array, N * sizeof(int));
	err =
		cudaMemcpy(dev_dim1_array, dim1_array, N * sizeof(int), cudaMemcpyHostToDevice);	
	if (err != cudaSuccess) {
		std::cerr << "WeightsErr is " << cudaGetErrorString(err) << std::endl;
	}
	err =
		cudaMemcpy(dev_dim2_array, dim2_array, N * sizeof(int), cudaMemcpyHostToDevice);	
	if (err != cudaSuccess) {
		std::cerr << "WeightsErr is " << cudaGetErrorString(err) << std::endl;
	}

//Dimensions preprocessing
	int maxN = *std::max_element(dim1_array, dim1_array+N);
	int maxM = *std::max_element(dim2_array, dim2_array+N);
	int ans_dim = maxN*maxM;
	double *ans_array = new double[ans_dim];
	for (int i = 0; i < ans_dim; i++){
		ans_array[i] = 0;
	}
	double **ans_ptr_array;
	ans_ptr_array = (double**)malloc(sizeof(double*));
	cudaMalloc((void**)&ans_ptr_array[0], ans_dim * sizeof(double));
	err =
		cudaMemcpy(ans_ptr_array[0], ans_array, ans_dim * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
		std::cerr << "1Err is " << cudaGetErrorString(err) << std::endl;
	}
	double **dev_ans_ptr_array;
	cudaMalloc(&dev_ans_ptr_array, sizeof(double*));
	err =
		cudaMemcpy(dev_ans_ptr_array, ans_ptr_array, sizeof(double*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}		

//Get results
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
//CUDA FUNCTION INVOKATION
	//weightedsum<<<GridSize(M), CU_BLOCK_SIZE>>>(dev_ptr_array, dev_ans_ptr_array, dev_weights_array, N, M);
	transpose<<<dim3(GridSize(maxN), GridSize(maxM), 1), dim3(CU_BLOCK_SIZE, CU_BLOCK_SIZE, 1)>>>(dev_ptr_array, dev_ans_ptr_array, dev_dim1_array, dev_dim2_array, N);
	//matrixproduct<<<dim3(GridSize(std::sqrt(M)), GridSize(std::sqrt(M)), 1), dim3(CU_BLOCK_SIZE, CU_BLOCK_SIZE, 1)>>>(dev_ptr_array, dev_ans_ptr_array, dev_dim1_array, dev_dim2_array, N, maxN, maxM);
//CUDA FUNCTION END
	
	end = std::chrono::system_clock::now();
	int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
                             (end-start).count();
    std::cout<<"COMPUTATION1 TIME IS: "<<elapsed_seconds<<std::endl;

//Copy results back
	err =
        cudaMemcpy(ans_ptr_array, dev_ans_ptr_array, sizeof(double *), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "6Err is " << cudaGetErrorString(err) << std::endl;
	}

//Print ans
	err =
	cudaMemcpy(ans_array, ans_ptr_array[0], ans_dim * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		std::cerr << "7Err is " << cudaGetErrorString(err) << std::endl;
	for (int i=0; i<ans_dim; i++){
		std::cout << ans_array[i] << " ";
		if (i > 0 & (i + 1) % maxM == 0) std::cout << std::endl;
	}
	std::cout << std::endl;

//Free memory
	for (int i = 0; i < N; i++){
		cudaFree(&host_ptr_array[i]);
	}
	cudaFree(&dev_ptr_array);
	cudaFree(&dev_ans_ptr_array);
	free(host_ptr_array);
	cudaDeviceReset();	
}

int main(int argc, char **argv){
	if (argc < 2){
		std::cout<<"Please enter N and M"<<std::endl;
		return 0;
	}
	//int N = atoi(argv[1]);
	int N = 1;
	int M = atoi(argv[1]);
	int K1 = atoi(argv[2]);
	int K2 = atoi(argv[3]);
	init(N, M, K1, K2);
	return 0;
}