#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm> 
#include <math.h>

const int CU_BLOCK_SIZE = 32;
#define GridSize(size) (size/CU_BLOCK_SIZE + 1)

//array - input(args)
//ans_array - output(rets)
//n, m - dimensions for each input matrix
//amount - amount of elements in each matrix
//maxn, maxm - maximum N and M from n and m accordingly
__global__
void matrixproduct(float** array, float** ans_array, int* n, int* m, int amount, int maxn, int maxm) {
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	if(Col < maxn && Row < maxm) { //filter unneeded threads
		int numCColumns = maxn;
		int numCRows = maxm;
		//if ((Row < m[0]) && (Col < n[0])){
		//	ans_array[0][Row*numCColumns + Col] = array[0][Row*numCColumns + Col]; //first matrix >> answer
			//printf("%f\n", ans_array[0][Row*numCColumns + Col]);
		//}
		int numARows, numAColumns, numBRows, numBColumns;
		for (int i = 0; i < amount; i++){
			numARows = numCRows; // A - left matrix
			numAColumns = numCColumns;
			numBRows = m[i]; // B - right matrix
			numBColumns = n[i];
			numCRows = numARows; // C - result matrix
			numCColumns = numBColumns;
			if ((Row < numARows) && (Col < numBColumns)){
    			float Cvalue = 0.0f;
    			for (int k = 0; k < numAColumns; ++k){
    				Cvalue += ans_array[0][Row*numAColumns + k] * array[i][k * numBColumns + Col];
    				printf("Col is %i, Row is %i, K is %i, %f x %f is %f\n",Col, Row, k, ans_array[0][Row*numAColumns + k], array[i][k * numBColumns + Col], ans_array[0][Row*numAColumns + k] * array[i][k * numBColumns + Col]);
    			}
    			ans_array[0][Row*numCColumns + Col] = Cvalue;
    		}
		}
	}
}

void init(int N, int M, int K1, int K2){
	cudaError_t err;
	cudaSetDevice(0);

	float *host_array = new float[M];
	std::cout<<"INPUT IS "<<N<<" VECTORS:"<<std::endl;
	for (int i = 0; i < M; i++){
		host_array[i] = i;
		std::cout<<host_array[i]<<" ";
	}
	std::cout<<std::endl;

	float **host_ptr_array;

	host_ptr_array = (float**)malloc(N*sizeof(float*));

//GPU memory allocation	
	for (int i = 0; i < N; i++){
		cudaMalloc((void**)&host_ptr_array[i], M * sizeof(float));
		err =
			cudaMemcpy(host_ptr_array[i], host_array, M * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cerr << "1Err is " << cudaGetErrorString(err) << std::endl;
		}
	}

	float **dev_ptr_array;
	cudaMalloc(&dev_ptr_array, N * sizeof(float*));
	err =
		cudaMemcpy(dev_ptr_array, host_ptr_array, N * sizeof(float*), cudaMemcpyHostToDevice);
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
	//int ans_dim = maxN*maxM;
	int maxdim = std::max(maxN, maxM);
	int ans_dim = maxdim*maxdim;
	float *ans_array = new float[ans_dim];
	for (int i = 0; i < maxdim; i++)
		for (int j = 0; j < maxdim; j++)
			i == j ? ans_array[i*maxdim + j] = 1 : ans_array[i*maxdim + j] = 0;

	float **ans_ptr_array;
	ans_ptr_array = (float**)malloc(sizeof(float*));
	cudaMalloc((void**)&ans_ptr_array[0], ans_dim * sizeof(float));
	err =
		cudaMemcpy(ans_ptr_array[0], ans_array, ans_dim * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
		std::cerr << "1Err is " << cudaGetErrorString(err) << std::endl;
	}
	float **dev_ans_ptr_array;
	cudaMalloc(&dev_ans_ptr_array, sizeof(float*));
	err =
		cudaMemcpy(dev_ans_ptr_array, ans_ptr_array, sizeof(float*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}		

//Get results
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
//CUDA FUNCTION INVOKATION
	//weightedsum<<<GridSize(M), CU_BLOCK_SIZE>>>(dev_ptr_array, dev_ans_ptr_array, dev_weights_array, N, M);
	matrixproduct<<<dim3(GridSize(maxdim), GridSize(maxdim), 1), dim3(CU_BLOCK_SIZE, CU_BLOCK_SIZE, 1)>>>(dev_ptr_array, dev_ans_ptr_array, dev_dim1_array, dev_dim2_array, N, maxN, maxM);
	//matrixproduct<<<dim3(GridSize(std::sqrt(M)), GridSize(std::sqrt(M)), 1), dim3(CU_BLOCK_SIZE, CU_BLOCK_SIZE, 1)>>>(dev_ptr_array, dev_ans_ptr_array, dev_dim1_array, dev_dim2_array, N, maxN, maxM);
//CUDA FUNCTION END
	
	end = std::chrono::system_clock::now();
	int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
                             (end-start).count();
    std::cout<<"COMPUTATION1 TIME IS: "<<elapsed_seconds<<std::endl;

//Copy results back
	err =
        cudaMemcpy(ans_ptr_array, dev_ans_ptr_array, sizeof(float *), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "6Err is " << cudaGetErrorString(err) << std::endl;
	}

//Print ans
	err =
	cudaMemcpy(ans_array, ans_ptr_array[0], ans_dim * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		std::cerr << "7Err is " << cudaGetErrorString(err) << std::endl;
	for (int i=0; i<ans_dim; i++){
		std::cout << ans_array[i] << " ";
		if (i > 0 & (i + 1) % maxN == 0) std::cout << std::endl;
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
	int K1 = atoi(argv[3]);
	int K2 = atoi(argv[4]);
	init(N, M, K1, K2);
	return 0;
}