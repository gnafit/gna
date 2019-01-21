if(GENERATE_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate=${CMAKE_BINARY_DIR}/profile-data")
    message(STATUS "Building with collection of PGO profiling enabled")
endif()

if(USE_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use=${CMAKE_BINARY_DIR}/profile-data -fprofile-correction")
    message(STATUS "Build using collected PGO profiling info")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++17-extensions")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebugInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ggdb3 -fno-omit-frame-pointer ")
endif()

if(UB_SANITIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebugInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()
