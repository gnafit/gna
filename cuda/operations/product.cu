#include <cuda.h>
#include <iostream>
#include <chrono>

const int CU_BLOCK_SIZE = 32;
#define GridSize(size) (size/CU_BLOCK_SIZE + 1)

__global__
void product(double** array, double** ans_array, int n, int m) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= m) return;
	ans_array[0][x] = array[0][x];
	for (int i = 1; i < n; i++){
		ans_array[0][x] *= array[i][x];
	}
}

void init(int N, int M){
	cudaError_t err;
	cudaSetDevice(0);

	double *host_array = new double[M];
	double *ans_array = new double[M];
	std::cout<<"INPUT IS "<<N<<" VECTORS:"<<std::endl;
	for (int i = 0; i < M; i++){
		host_array[i] = i;
		ans_array[i] = 0;
		std::cout<<host_array[i]<<" ";
	}
	std::cout<<std::endl;

	double **host_ptr_array;
	double **ans_ptr_array;

	host_ptr_array = (double**)malloc(N*sizeof(double*));
	ans_ptr_array = (double**)malloc(N*sizeof(double*));

//GPU memory allocation	
	for (int i = 0; i < N; i++){
		cudaMalloc((void**)&host_ptr_array[i], M * sizeof(double));
		err =
			cudaMemcpy(host_ptr_array[i], host_array, M * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "1Err is " << cudaGetErrorString(err) << std::endl;
		}
		if (i < N){
			cudaMalloc((void**)&ans_ptr_array[i], M * sizeof(double));
			err =
				cudaMemcpy(ans_ptr_array[i], ans_array, M * sizeof(double), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cerr << "1Err is " << cudaGetErrorString(err) << std::endl;
			}		
		}
	}

	double **dev_ptr_array;
	double **dev_ans_ptr_array;
	cudaMalloc(&dev_ptr_array, N * sizeof(double*));
	cudaMalloc(&dev_ans_ptr_array, N * sizeof(double*));
	err =
		cudaMemcpy(dev_ptr_array, host_ptr_array, N * sizeof(double*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}
	err =
		cudaMemcpy(dev_ans_ptr_array, ans_ptr_array, N * sizeof(double*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}		


//Get results
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

//CUDA FUNCTION INVOKATION
	product<<<GridSize(M), CU_BLOCK_SIZE>>>(dev_ptr_array, dev_ans_ptr_array, N, M);
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
	err =
	cudaMemcpy(ans_array, ans_ptr_array[0], M * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		std::cerr << "6Err is " << cudaGetErrorString(err) << std::endl;
	for (int i=0; i<M; i++){
		std::cout << ans_array[i] << " ";
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
	int N = atoi(argv[1]);
	int M = atoi(argv[2]);
	init(N, M);
	return 0;
}