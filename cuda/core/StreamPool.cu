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
		cudaStreamCreateWithFlags(&streampool[i], cudaStreamNonBlocking);
	}
	std::cerr << "StreamPool Created" << std::endl;
}

~StreamPool::StreamPool {

	for (int i = 0; i < CUDA_STREAMS_NUM; ++i) {
		cudaStreamDestroy(&streampool[i])
	}
	std::cerr << "Stream Pool killed " << std::endl;
}

size_t StreamPool::getFreeStreamId () {
	size_t index = -1;
	do {
		index = (index + 1) % CUDA_STREAMS_NUM;
	} while (cudaStreamQuery(&streampool[index]) == cudaErrorNotReady);

	if (cudaErrorInvalidResourceHandle == cudaStreamQuery(&streampool[index])) {
		std::cerr << "Stream " << index << " is broken!" << std::endl;
		// TODO Throw error ?
	}
	return index;
}

cudaStream_t& StreamPool::getStream(size_t index) {
	// TODO throw error if index is wrong?
	return &streampool[index];
}
