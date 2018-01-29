project(cu)

                                                                                                                                                                              
cmake_minimum_required(VERSION 2.8)
                                                                                                                                                                                                
set(CMAKE_MODULE_PATH cmake)

#find_package( CUDA REQUIRED)

#set(CUDA_HOST_COMPILER "clang++")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50 -std=c++11 -Xcompiler -fPIC )

include_directories(extra/ core/)

set(CUSOURCES
  extra/GNAcuGpuArray.cu
  extra/GNAcuMath.cu
  extra/GNAcuOscProbFull.cu
  extra/GNAcuOscProbMem.cu
  )

set(CUHEADERS 
  extra/GNAcuGpuArray.hh
  extra/GNAcuMath.h
  extra/GNAcuOscProbFull.h
  extra/GNAcuOscProbMem.hh
#  extra/GNAcuRootMath.h
  )

configure_file( ${CMAKE_BINARY_DIR}/../config/gna/cuda_config_vars.h.in ${CMAKE_BINARY_DIR}/generated/cuda_config_vars.h)
include_directories( SYSTEM ${CMAKE_BINARY_DIR}/generated/ )

cuda_add_library(cuGNA SHARED ${CUSOURCES} ${CUHEADERS})
cuda_add_cublas_to_target(cuGNA)

