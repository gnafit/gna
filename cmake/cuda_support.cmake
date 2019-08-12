#
# Configure CUDA specific options
#

#configure_file(core/config_vars.h.in ${CMAKE_BINARY_DIR}/generated/config_vars.h)

if(CUDA_SUPPORT)
    message(STATUS "CUDA is enabled")
    find_package(CUDA REQUIRED)
    include_directories(cuda/operations/ cuda/extra/ cuda/core/)
    if(NOT CUDA_MAT_SIZE_THRESHOLD)
        set(CUDA_MAT_SIZE_THRESHOLD 100)
    endif()
#    message(STATUS "CUDA_MAT_SIZE_THRESHOLD = ${CUDA_MAT_SIZE_THRESHOLD}")
    set(CUDA_ADD_LIBS)
    IF (CUDA_FOUND)
        add_subdirectory(cuda/)
        #set (CUDA_ADD_LIBS ${CUDA_CUBLAS_LIBRARIES} -lgslcblas cuGNA)
        set (CUDA_ADD_LIBS cuGNA)
        link_directories(cuda/)
    ENDIF()
    set (CUSOURCES
        )
    set (CUHEADERS
         GpuArray.hh
         GpuArrayTypes.hh
         cuInterpExpo.hh
         GpuBasics.hh
         cuElementary.hh
         cuOscProbPMNS.hh
         cuIntegrate.hh
        )
else()
    message(STATUS "CUDA is disabled")
    set(CUDA_SUPPORT 0)
    set(CUSOURCES "")
    set(CUHEADERS "")
endif()

if (NOT CUDA_DEBUG_INFO)
    set(CUDA_DEBUG_INFO 0)
endif()

