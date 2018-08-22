#include "StreamPool.hh"


StreamPool::StreamPool {
	if (streampool) {
		std::cerr << "Double cuda streams pool creation! Smth wrong!" << std::endl;
		return;
	}
	if (CUDA_STREAMS_NUM < 1) {
		std::cerr << "Can't create " << CUDA_STREAMS_NUM 
				<< " streams! You should set CUDA_STREAMS_NUM!" << std::endl;
	}
	cudaStream_t* streampool = new cudaStream_t[CUDA_STREAMS_NUM];
	for (size_t i = 0; i < CUDA_STREAMS_NUM; ++i) {
		cudaStreamCreate(&streampool[i]);
	}
	std::cerr << "StreamPool Created" << std::endl;
}

~StreamPool::StreamPool {
	std::cerr << "Stream Pool killed " << std::endl;
}
