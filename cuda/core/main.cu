#include "GpuBasics.hh"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


#define TYP double


template<typename T>
__global__ void inverseglob(T* in, T* out) {
	inverse(in, out);
//	prodNumToVec(1, in, out);
}

int main() {
	cudaError_t err;

	int N = 100, M=100;
	TYP *host_array = new TYP[N];
        std::cout<<"INPUT IS "<<N<<" VECTORS:"<<std::endl;
        for (int i = 0; i < M; i++){
                host_array[i] = TYP(i);
                std::cout<<host_array[i]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<std::endl;

	
	TYP *dev_ptr_array;
	TYP *dev_ans;

	err = cudaMalloc(&dev_ptr_array, N * sizeof(TYP*));
	err = cudaMalloc(&dev_ans, N * sizeof(TYP*));
	err = cudaMemcpy(dev_ptr_array, host_array, N * sizeof(TYP*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}

//	inverseglob<<<1,N>>>(dev_ptr_array, dev_ans);

	copyH2D_NA(dev_ans, host_array, N);
	debug_drop(dev_ans+40, N-41);
	cudaDeviceSynchronize();

	
	err = cudaMemcpy( host_array, dev_ans, N * sizeof(TYP*), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "2Err is " << cudaGetErrorString(err) << std::endl;
	}

/*
        for (int i = 0; i < M; i++){
                std::cout<<host_array[i]<<" ";
        }
        std::cout<<std::endl;
*/
	return 0;
}
