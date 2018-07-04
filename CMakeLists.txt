cmake_minimum_required(VERSION 3.0)

project(GlobalNuAnalysis2)
set(LIBNAME GlobalNuAnalysis2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

enable_language(Fortran)

set (SOURCES
  core/GNAObject.cc
  core/ParametersGroup.cc
  core/Parametrized.cc
  core/TransformationDescriptor.cc
  core/UncertainParameter.cc
  core/transformation/Accessor.cc
  core/transformation/Args.cc
  core/transformation/Atypes.cc
  core/transformation/EntryHandle.cc
  core/transformation/OutputHandle.cc
  core/transformation/InputHandle.cc
  core/transformation/Initializer.hh
  core/transformation/Rets.cc
  core/transformation/Rtypes.cc
  core/transformation/Source.cc
  core/transformation/TransformationBase.cc
  core/transformation/TransformationEntry.cc
  core/transformation/TransformationErrors.cc
  core/transformation/TypesFunctions.cc
  transformations/base/Prediction.cc
  transformations/base/Product.cc
  transformations/base/Sum.cc
  transformations/base/WeightedSum.cc
  transformations/calculus/Derivative.cc
  transformations/detector/C14Spectrum.cc
  transformations/detector/EnergyResolution.cc
  transformations/detector/HistNonlinearity.cc
  transformations/detector/HistSmear.cc
  transformations/functions/SelfPower.cc
  transformations/functions/Exp.cc
  transformations/hist/Rebin.cc
  transformations/integrator/GaussLegendre.cc
  transformations/integrator/GaussLegendre2d.cc
  transformations/interpolation/LinearInterpolator.cc
  transformations/interpolation/SegmentWise.cc
  transformations/interpolation/InterpExpo.cc
  transformations/linalg/Normalize.cc
  transformations/linalg/RenormalizeDiag.cc
  transformations/neutrino/IbdFirstOrder.cc
  transformations/neutrino/IbdInteraction.cc
  transformations/neutrino/IbdZeroOrder.cc
  transformations/neutrino/OscProb2nu.cc
  transformations/neutrino/OscProbMatterUniform.cc
  transformations/neutrino/OscProbPMNS.cc
  transformations/neutrino/OscProbPMNSDecoh.cc
  transformations/neutrino/ReactorGroup.cc
  transformations/neutrino/ReactorNorm.cc
  transformations/stats/Chi2.cc
  transformations/stats/CovarianceToyMC.cc
  transformations/stats/CovariatedPrediction.cc
  transformations/stats/Covmat.cc
  transformations/stats/NormalToyMC.cc
  transformations/stats/Poisson.cc
  transformations/stats/PoissonToyMC.cc
  transformations/var/VarDiff.cc
  transformations/var/VarProduct.cc
  transformations/var/VarSum.cc
  transformations/var/VarArray.cc
  extra/GridFilter.cc
  )

set (HEADERS
  # core/
  Data.hh
  Random.hh
  GeoNeutrinoFluxNormed.hh
  Exceptions.hh
  # core/transformation/
  Sink.hh
  Source.hh
  Args.hh
  Rets.hh
  Rtypes.hh
  Atypes.hh
  TypesFunctions.hh
  EntryHandle.hh
  OutputHandle.hh
  OpenHandle.hh
  InputHandle.hh
  TransformationDebug.hh
  TransformationBase.hh
  TransformationEntry.hh
  TransformationErrors.hh
  TransformationDescriptor.hh
  TransformationBind.hh
  SingleOutput.hh
  Initializer.hh
  Accessor.hh
  # core/
  GNAObject.hh
  UncertainParameter.hh
  # transformations/
  Prediction.hh
  CovariatedPrediction.hh
  Product.hh
  Sum.hh
  WeightedSum.hh
  Points.hh
  Histogram.hh
  HistEdges.hh
  Identity.hh
  Dummy.hh
  LinearInterpolator.hh
  SegmentWise.hh
  InterpExpo.hh
  FillLike.hh
  Derivative.hh
  Covmat.hh
  Cholesky.hh
  Chi2.hh
  Poisson.hh
  Exp.hh
  SelfPower.hh
  Normalize.hh
  CovarianceToyMC.hh
  NormalToyMC.hh
  PoissonToyMC.hh
  Rebin.hh
  VarArray.hh
  VarProduct.hh
  VarSum.hh
  VarDiff.hh
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
  # examples/
  GaussianPeakWithBackground.hh
  # extra/
  GridFilter.hh
  EigenHelpers.hh
  )

include_directories(core/transformation/
                    core/
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
                    extra/
                    examples/
                   )

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

ROOT_GENERATE_DICTIONARY(Dict "${HEADERS}" OPTIONS -interpreteronly)

add_library(${LIBNAME} SHARED ${SOURCES} Dict.cxx ${CONTRIB_SRC})
target_link_libraries(${LIBNAME} ${ROOT_LIBRARIES} -lPyROOT -lgsl -lgslcblas -lm)
