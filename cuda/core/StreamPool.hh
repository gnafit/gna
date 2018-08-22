#pragma once

#include <cuda.h>
#include <iostream>

#include "cuda_config_vars.h"

class StreamPool {
public:
	StreamPool();
	~StreamPool();

	cudaStream_t* streampool = nullptr;
	
};
