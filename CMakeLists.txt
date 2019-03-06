cmake_minimum_required(VERSION 3.5)

project(GlobalNuAnalysis2)
set(LIBNAME GlobalNuAnalysis2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(core)
add_subdirectory(transformations)
add_subdirectory(extra)
add_subdirectory(examples)
add_subdirectory(unittest)
add_subdirectory(thirdparty/chengyp/)

set(SOURCES ${CORE_SOURCES} ${TRANSFORMATION_SOURCES} ${EXTRA_SOURCES} ${EXAMPLES_SOURCES} ${UNITTEST_SOURCES})
set(HEADERS ${CORE_HEADERS} ${TRANSFORMATION_HEADERS} ${EXTRA_HEADERS} ${EXAMPLES_HEADERS} ${UNITTEST_HEADERS})

include_directories(core/parameters/
                    core/transformation/
                    core/
                    transformations/trial/
                    transformations/backgrounds/
                    transformations/base/
                    transformations/calculus/
                    transformations/detector/
                    transformations/functions/
                    transformations/hist/
                    transformations/integrator/
                    transformations/interpolation/
                    transformations/linalg/
                    transformations/neutrino/
                    transformations/stats/
                    transformations/var/
                    transformations/debug/
                    transformations/legacy/
                    extra/
                    examples/
                    unittest/
                    thirdparty/chengyp/
                   )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Setting ${CMAKE_BUILD_TYPE} build")

option(GENERATE_PROFILE OFF)
option(USE_PROFILE OFF)
option(UB_SANITIZE OFF)
option(LTO OFF)

#Check if LTO is possible
if(LTO)
    if(POLICY CMP0069)
        cmake_policy(SET CMP0069 NEW)
    endif()
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
endif()

set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)
message(STATUS "Compiling in C++${CMAKE_CXX_STANDARD} mode")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Werror -Wall  -pedantic -pipe")

include(compilation_opts)
include(debug_opts)

add_subdirectory(fmt)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "$ENV{ROOTSYS}/etc/cmake")
# list(APPEND CMAKE_PREFIX_PATH "$ENV{ROOTSYS}")

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(Boost 1.40.0 REQUIRED)
if(Boost_FOUND)
    message(STATUS "found Boost version ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION} in ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
    message(FATAL_ERROR "Boost >= 1.40.0 not found")
endif(Boost_FOUND)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
include_directories(SYSTEM "fmt")

find_package(GSL REQUIRED)
find_package(ROOT REQUIRED)
# include(${ROOT_USE_FILE})

link_directories(${ROOT_LIBRARY_DIR})

ROOT_GENERATE_DICTIONARY(Dict "${HEADERS}" OPTIONS -interpreteronly)

add_library(${LIBNAME} SHARED ${SOURCES} Dict.cxx ${CONTRIB_SRC})

if(result)
    set_property(TARGET ${LIBNAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

target_include_directories(${LIBNAME} SYSTEM PRIVATE ${ROOT_INCLUDE_DIR})
target_include_directories(${LIBNAME} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(${LIBNAME} PRIVATE  ${ROOT_LIBRARIES} -lPyROOT)
target_link_libraries(${LIBNAME} PRIVATE GSL::gsl GSL::gslcblas)
target_link_libraries(${LIBNAME} PRIVATE fmt)

if(GENERATE_PROFILE)
    target_link_libraries(${LIBNAME} PRIVATE "-fprofile-generate=${CMAKE_BINARY_DIR}/profile-data")
endif()

if(UB_SANITIZE)
    target_link_libraries(${LIBNAME} PRIVATE "-fsanitize=undefined")
endif()
