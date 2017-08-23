project(cu)

                                                                                                                                                                              
cmake_minimum_required(VERSION 2.8)
                                                                                                                                                                                                
set(CMAKE_MODULE_PATH cmake)

find_package( CUDA REQUIRED)

#set(CUDA_HOST_COMPILER "clang++")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50 -std=c++11 -Xcompiler -fPIC)
                                                                                                                                                                          
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50)

cuda_add_library(GNAcuMath SHARED GNAcuMath.cu GNAcuMath.h)
cuda_add_cublas_to_target(GNAcuMath)

# old example
#cuda_add_library(simp SHARED simp.cu simp.h)
#cuda_add_cublas_to_target(simp)
