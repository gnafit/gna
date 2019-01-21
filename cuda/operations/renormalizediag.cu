#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm> 
#include <math.h>

const int CU_BLOCK_SIZE = 32;
#define GridSize(size) (size/(CU_BLOCK_SIZE) + 1)

__device__
double vectorsumcolumn(double** array, int n, int x){
	double del = 0;
	for (int j = 0; j < n; j++){
		del += array[0][j*n + x];
	}
	return del==0.0 ? 1.0 : del;
}

__global__
void renormalizediag(double** array, double** ans_array, int* n, int* m, int Diag, int multiplier) { //NOTE: Normalize and normalize_segment can lead to cudaErrorLaunchTimeout on big amount of elements due to watchdog!
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if (x < n[0] & y < n[0]){
		ans_array[0][y*n[0]+x] = array[0][y*n[0]+x];
		if (abs(x - y) <= Diag - 1) //multiply diagonals
			ans_array[0][y*n[0] + x] *= multiplier;
		__syncthreads();
		//printf("%i %i\n", x, y);
   		ans_array[0][y*n[0] + x] = ans_array[0][y*n[0] + x]/vectorsumcolumn(ans_array, n[0], x);
	}
}

void init(int N, int M, int K1, int K2, int Diag){
	cudaError_t err;
	cudaSetDevice(0);

	double *host_array = new double[M];
	std::cout<<"INPUT IS "<<N<<" VECTORS:"<<std::endl;
	for (int i = 0; i < M; i++){
		host_array[i] = i+1;
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

	renormalizediag<<<dim3(GridSize(M), GridSize(M), 1), dim3(CU_BLOCK_SIZE, CU_BLOCK_SIZE, 1)>>>(dev_ptr_array, dev_ans_ptr_array, dev_dim1_array, dev_dim2_array, Diag, 2);
	//normalize_segment<<<GridSize(M), CU_BLOCK_SIZE>>>(dev_ptr_array, dev_ans_ptr_array, dev_dim1_array, 0, 2);
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
	for (int j = 0; j < N; j++){
	err =
	cudaMemcpy(ans_array, ans_ptr_array[j], ans_dim * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		std::cerr << "7Err is " << cudaGetErrorString(err) << std::endl;
	for (int i = 0; i < M; i++)
		std::cout << ans_array[i] << " ";
	std::cout << std::endl;
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
	int N = 1; //Amount of input arrays(always 1 here)
	int M = atoi(argv[1]); //Amount of elements in each input array
	int K1 = atoi(argv[2]); //Matrix Dimension X
	int K2 = atoi(argv[3]); //Matrix Dimension Y
	if (K1 != K2){
		std::cout<<"Matrix isnt square!"<<std::endl;
		return 0;
	}
	int Diag = atoi(argv[4]); //Number of diagonals to scale
	init(N, M, K1, K2, Diag);
	return 0;
}