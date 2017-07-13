set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14 -Wno-deprecated-declarations -Werror -Wall  -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pipe")

if(NATIVE)
    message(STATUS "Using native set of CPU instructions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if(TRANS_DEBUG)
    message(STATUS "Enabling debug printing for transformations")
    add_definitions(-DTRANSFORMATION_DEBUG)
endif()

if(PARAM_DEBUG)
    message(STATUS "Enabling debug printing for parameters")
    add_definitions(-DDEBUG_PARAMETERS)
endif()

if(GRIDFILTER_DEBUG)
    message(STATUS "Setting debug mode")
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDEBUG_GRIDFILTER)
endif()

