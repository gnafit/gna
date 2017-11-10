cmake_minimum_required(VERSION 3.0)

project(GlobalNuAnalysis2)
set(LIBNAME GlobalNuAnalysis2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

enable_language(Fortran)

set (SOURCES
  core/TransformationBase.cc
  core/Transformation.cc
  core/Parametrized.cc
  core/GNAObject.cc
  core/UncertainParameter.cc
  core/ParametersGroup.cc
  core/Prediction.cc
  core/CovariatedPrediction.cc
  core/Product.cc
  core/Sum.cc
  core/WeightedSum.cc
  core/LinearInterpolator.cc
  core/Derivative.cc
  core/Covmat.cc
  core/Chi2.cc
  core/Poisson.cc
  core/CovarianceToyMC.cc
  core/NormalToyMC.cc
  core/PoissonToyMC.cc
  core/Rebin.cc
  neutrino/OscProb2nu.cc
  neutrino/OscProbPMNS.cc
  neutrino/OscProbPMNSDecoh.cc
  neutrino/IbdInteraction.cc
  neutrino/IbdZeroOrder.cc
  neutrino/IbdFirstOrder.cc
  neutrino/ReactorNorm.cc
  neutrino/ReactorGroup.cc
  neutrino/OscProbMatterUniform.cc
  integrator/GaussLegendre.cc
  integrator/GaussLegendre2d.cc
  detector/EnergyResolution.cc
  detector/HistNonlinearity.cc
  detector/HistSmear.cc
  detector/RenormalizeDiag.cc
  detector/C14Spectrum.cc
  extra/GridFilter.cc
  )

set (HEADERS
  Random.hh
  GeoNeutrinoFluxNormed.hh
  Exceptions.hh
  Transformation.hh
  GNAObject.hh
  UncertainParameter.hh
  Prediction.hh
  CovariatedPrediction.hh
  Product.hh
  Sum.hh
  WeightedSum.hh
  Points.hh
  Histogram.hh
  HistEdges.hh
  Identity.hh
  LinearInterpolator.hh
  FillLike.hh
  Derivative.hh
  Covmat.hh
  Cholesky.hh
  Chi2.hh
  Poisson.hh
  CovarianceToyMC.hh
  NormalToyMC.hh
  PoissonToyMC.hh
  Rebin.hh
  Minimizable.hh
  Neutrino.hh
  OscillationVariables.hh
  PMNSVariables.hh
  OscProb2nu.hh
  OscProbPMNS.hh
  OscProbPMNSDecoh.hh
  OscProbMatterUniform.hh
  EvisToEe.hh
  IbdInteraction.hh
  IbdZeroOrder.hh
  IbdFirstOrder.hh
  ReactorNorm.hh
  ReactorGroup.hh
  GaussLegendre.hh
  GaussLegendre2d.hh
  EnergyResolution.hh
  HistNonlinearity.hh
  HistSmear.hh
  RenormalizeDiag.hh
  C14Spectrum.hh
  GaussianPeakWithBackground.hh
  GridFilter.hh
  EigenHelpers.hh
  )

include_directories(core/ extra/ neutrino/ integrator/ detector/ examples/ backgrounds/)
set(DICT    Dict.cc)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Setting ${CMAKE_BUILD_TYPE} build")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)
message(STATUS "Compiling in C++${CMAKE_CXX_STANDARD} mode")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Werror -Wall  -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")
include(dev-tools)



set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "$ENV{ROOTSYS}/etc/cmake")

find_package( Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package( Boost 1.40.0 REQUIRED)
if(Boost_FOUND)
    message(STATUS "found Boost version ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION} in ${Boost_INCLUDE_DIRS}")
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
else(Boost_FOUND)
    message(FATAL_ERROR "Boost >= 1.40.0 not found")
endif(Boost_FOUND)

find_Package(GSL REQUIRED)
include_directories(SYSTEM ${GSL_INCLUDE_DIR})

find_package(ROOT)
include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
link_directories(${ROOT_LIBRARY_DIR})

ROOT_GENERATE_DICTIONARY(Dict "${HEADERS}" LINKDEF LinkDef.h)

add_library(${LIBNAME} SHARED ${SOURCES} Dict.cxx ${CONTRIB_SRC})
target_link_libraries(${LIBNAME} ${ROOT_LIBRARIES} -lPyROOT -lgsl -lgslcblas -lm)
