cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package( CUDA REQUIRED)
message(STATUS "Cuda cublas libraries  : ${CUDA_CUBLAS_LIBRARIES}")

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11 -Xcompiler -fPIC  -O3 -arch=compute_50 
)

include_directories(${CUDA_INCLUDE} ${CUDA_INCLUDE_DIRS})
set(LINK_DIRECTORIES
${CUDA_LIBRARIES}
)
cuda_compile(GNAtestCUDA_O test.cu)
cuda_add_library(CNAtestCUDA SHARED test.h ${GNAtestCUDA_O})
target_link_libraries(CNAtestCUDA ${CUDA_LIBRARY} -lcuda )
install(TARGETS CNAtestCUDA DESTINATION ${CMAKE_BINARY_DIR}/lib)
