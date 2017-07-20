project(cu)

                                                                                                                                                                              
cmake_minimum_required(VERSION 2.8)
                                                                                                                                                                                                
set(CMAKE_MODULE_PATH ../cmake)

find_package( CUDA REQUIRED)

                                                                                                                                                                           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50)

cuda_add_library(simp SHARED simp.cu simp.h)
