#pragma once

#include <cuda.h>
#include <iostream>

#include "cuda_config_vars.h"

class StreamPool {
public:
	StreamPool();
	~StreamPool();

	size_t getFreeStreamId();

	cudaStream_t& getStream(size_t index);
private
	cudaStream_t* streampool = nullptr;
	
};
