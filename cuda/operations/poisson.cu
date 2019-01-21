#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm> 
#include <math.h>

const int CU_BLOCK_SIZE = 64;
#define GridSize(size) (size/CU_BLOCK_SIZE + 1)

__device__ 
double factorial(double n)
{
    if (n < 2)
        return 1; 
    return n*factorial(n - 1);
}

__device__
void vectorsum(double** array, int n, int i){
	for (int j = 1; j < n; j++)
		array[i/2][0] += array[i/2][j];
	array[i/2][0] *= -2;
}

__global__
void poisson(double** array, double** ans_array, int* n, int amount, int maxn) {
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	
	if(Col < maxn) { //filter unneeded threads
		for (int i = 0; i < amount-1; i+=2){
			//if (n[i]!= n[i+1]) printf("Dimensions do not correspond!\n");
			if (Col < n[i]){
				double fact = factorial(array[i+1][Col]);
				__syncthreads();
    			ans_array[i/2][Col] += log(array[i][Col]) * array[i+1][Col] - array[i][Col] - log(fact);
			if (Col == 0)
	    			vectorsum(ans_array, n[i], i);
    		}
		}
	}
}

__global__
void poissonapprox(double** array, double** ans_array, int* n, int amount, int maxn) {
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	
	if(Col < maxn) { //filter unneeded threads
		for (int i = 0; i < amount-1; i+=2){
			//if (n[i]!= n[i+1]) printf("Dimensions do not correspond!\n");
			if (Col < n[i]){
				__syncthreads();
    			ans_array[i/2][Col] += log(array[i][Col]) * array[i+1][Col] - array[i][Col] - array[i+1][Col]*log(array[i+1][Col]);
			if (Col == 0)
	    			vectorsum(ans_array, n[i], i);
    		}
		}
	}
}

void init(int N, int M, int K1, int K2){
	cudaError_t err;
	cudaSetDevice(0);

	double *host_array = new double[M];
	std::cout<<"INPUT IS "<<N<<" VECTORS:"<<std::endl;
	for (int i = 0; i < M; i++){
		host_array[i] = double(i+1)/(10000);
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
		dim1_array[i] = K1;
		dim2_array[i] = K2;
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
	ans_ptr_array = (double**)malloc(N * sizeof(double*));
	for (int i = 0; i < N; i++){
		cudaMalloc((void**)&ans_ptr_array[i], ans_dim * sizeof(double));
		err =
			cudaMemcpy(ans_ptr_array[i], ans_array, ans_dim * sizeof(double), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
			std::cerr << "1Err is " << cudaGetErrorString(err) << std::endl;
		}
	}
	double **dev_ans_ptr_array;
	cudaMalloc(&dev_ans_ptr_array, N * sizeof(double*));
	err =
		cudaMemcpy(dev_ans_ptr_array, ans_ptr_array, N * sizeof(double*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}		

//Get results
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
//CUDA FUNCTION INVOKATION

	poisson<<<GridSize(M), CU_BLOCK_SIZE>>>(dev_ptr_array, dev_ans_ptr_array, dev_dim1_array, N, maxN);
	
//CUDA FUNCTION END
	
	end = std::chrono::system_clock::now();
	int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
                             (end-start).count();
    std::cout<<"COMPUTATION1 TIME IS: "<<elapsed_seconds<<std::endl;

//Copy results back
	err =
        cudaMemcpy(ans_ptr_array, dev_ans_ptr_array, N * sizeof(double *), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "6Err is " << cudaGetErrorString(err) << std::endl;
	}

//Print ans
	for (int j = 0; j < N/2; j++){
	err =
	cudaMemcpy(ans_array, ans_ptr_array[j], ans_dim * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		std::cerr << "7Err is " << cudaGetErrorString(err) << std::endl;
	//for (int i = 0; i < M; i++){
	std::cout << ans_array[0] << std::endl; // First element of first N/2 rets vectors contains poisson value
}

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
	int N = atoi(argv[1]); //Amount of input arrays
	int M = atoi(argv[2]); //Amount of elements in each input array
	int K1 = M; //Matrix Dimension X (always M here)
	int K2 = 1; //Matrix Dimension Y (always 1 here)
	init(N, M, K1, K2);
	return 0;
}
