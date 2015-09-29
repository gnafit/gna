cmake_minimum_required(VERSION 2.8)

project(GlobalNuAnalysis2)
set(LIBNAME GlobalNuAnalysis2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

enable_language(Fortran)

set (SOURCES
  core/TransformationBase.cc
  core/Transformation.cc
  core/Parametrized.cc
  core/GNAObject.cc
  core/PredictionSet.cc
  core/UncertainParameter.cc
  core/ParametersGroup.cc
  neutrino/OscProb2nu.cc
  integrator/GaussLegendre.cc
  )

set (HEADERS
  GNAObject.hh
  UncertainParameter.hh
  PredictionSet.hh
  PointSet.hh
  OscProb2nu.hh
  GaussLegendre.hh
  )
  
include_directories(core/ neutrino/ integrator/)
set(DICT    Dict.cc)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wno-deprecated-declarations -Wall -pedantic -Werror")

list(APPEND CMAKE_MODULE_PATH "$ENV{ROOTSYS}/etc/cmake")

find_package( Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package( Boost 1.40.0 )
if(Boost_FOUND)
    message(STATUS "found Boost version ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION} in ${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
else(Boost_FOUND)
    message(FATAL_ERROR "Boost >= 1.40.0 not found")
endif(Boost_FOUND)

find_Package( GSL REQUIRED )
include_directories(${GSL_INCLUDE_DIR})

find_package(ROOT)
include_directories(${ROOT_INCLUDE_DIR})
link_directories(${ROOT_LIBRARY_DIR})

ROOT_GENERATE_DICTIONARY(Dict "${HEADERS}" LINKDEF LinkDef.h)

add_library(${LIBNAME} SHARED ${SOURCES} Dict.cxx ${CONTRIB_SRC})
target_link_libraries(${LIBNAME} ${ROOT_LIBRARIES} -lMathMore -lPyROOT)
