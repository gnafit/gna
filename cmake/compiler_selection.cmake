#
# Use COMPILER option to quickly switch between gcc/clang
#
if(COMPILER)
    if(COMPILER STREQUAL "gcc")
        message(STATUS "Using gcc/g++ compilers")
        set(CMAKE_CXX_COMPILER, "g++")
        set(CMAKE_C_COMPILER, "gcc")
    elseif(COMPILER STREQUAL "clang")
        message(STATUS "Using clang/clang++ compilers")
        set(CMAKE_CXX_COMPILER, "clang++")
        set(CMAKE_C_COMPILER, "clang")
    else()
        message(FATAL_ERROR "Option COMPILER is set, but the value is unsupported (${COMPILER})")
    endif()
else()
    message(STATUS "Using system cc/c++ compilers")
endif()

