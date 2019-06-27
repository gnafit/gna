project(cu)

                                                                                                                                                                              
cmake_minimum_required(VERSION 2.8)
                                                                                                                                                                                                
set(CMAKE_MODULE_PATH cmake)

#find_package( CUDA REQUIRED)

#set(CUDA_HOST_COMPILER "clang++")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_COMPUTING_CAPABILITY 52)

execute_process(COMMAND bash -c "lspci | sed -r 's/^.*NVIDIA.*\\[(.*)\\].*$/\\1/;t;d' | head -n 1 | xargs echo -n" OUTPUT_VARIABLE CUDA_GPU_CARD)
if(${CUDA_GPU_CARD} STREQUAL "GeForce GTX 950M")
    set(CUDA_COMPUTING_CAPABILITY 50)
#elseif(${CUDA_GPU_CARD} STREQUAL "GeForce GTX 950M")
    #set(CUDA_COMPUTING_CAPABILITY 50)
endif()

message(STATUS "GPU: "${CUDA_GPU_CARD})
message(STATUS "Computing capability: "${CUDA_COMPUTING_CAPABILITY})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_${CUDA_COMPUTING_CAPABILITY},code=sm_${CUDA_COMPUTING_CAPABILITY} -std=c++11 -Xcompiler -fPIC )

include_directories(extra/ core/)
configure_file( cuda_config_vars.h.in ${CMAKE_BINARY_DIR}/generated/cuda_config_vars.h)
include_directories( ${CMAKE_BINARY_DIR}/generated/ )

set(CUSOURCES
  core/GpuBasics.cu
  core/GpuArray.cu
  extra/cuInterpExpo.cu
  operations/sum.cu
  operations/weightedsum.cu
  operations/product.cu
  operations/identity.cu
  operations/exp.cu
  operations/selfpower.cu
  operations/poisson.cu
  operations/fillike.cu
  operations/normalize.cu
  extra/cuOscProbPMNS.cu
  )

set(CUHEADERS
  cuda_config_vars.h 
  operations/cuOperations.cuh
  core/GpuBasics.hh
  core/GpuArray.hh 
  core/GpuArrayTypes.hh
  extra/cuRootMath.h
  extra/cuInterpExpo.hh
  core/cuElementary.hh
  extra/cuOscProbPMNS.hh
  )

cuda_add_library(cuGNA SHARED ${CUSOURCES} ${CUHEADERS})
#cuda_add_cublas_to_target(cuGNA)

