cmake_minimum_required(VERSION 3.0)

project(GlobalNuAnalysis2)
set(LIBNAME GlobalNuAnalysis2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set (SOURCES
  core/transformation/Accessor.cc
  core/transformation/Args.cc
  core/transformation/Atypes.cc
  core/transformation/EntryHandle.cc
  core/transformation/OutputHandle.cc
  core/transformation/InputHandle.cc
  core/transformation/Initializer.hh
  core/transformation/Rets.cc
  core/transformation/Ints.cc
  core/transformation/Rtypes.cc
  core/transformation/Itypes.cc
  core/transformation/Source.cc
  core/transformation/TransformationBase.cc
  core/transformation/TransformationEntry.cc
  core/transformation/TransformationErrors.cc
  core/transformation/TypesFunctions.cc
  core/GNAObject.cc
  core/ParametersGroup.cc
  core/Parametrized.cc
  core/TransformationDescriptor.cc
  core/UncertainParameter.cc
  transformations/trial/TrialMultiFunc.cc
  transformations/base/FillLike.cc
  transformations/base/Histogram.cc
  transformations/base/Histogram2d.cc
  transformations/base/Identity.cc
  transformations/base/Points.cc
  transformations/base/Bins.cc
  transformations/base/Concat.cc
  transformations/base/Product.cc
  transformations/base/Sum.cc
  transformations/base/Ratio.cc
  transformations/base/WeightedSum.cc
  transformations/calculus/Derivative.cc
  transformations/calculus/Jacobian.cc
  transformations/detector/FakeGSLFunction.cc
  transformations/detector/C14Spectrum.cc
  transformations/detector/EnergyResolution.cc
  transformations/detector/HistNonlinearity.cc
  transformations/detector/HistSmear.cc
  transformations/detector/HistSmearSparse.cc
  transformations/functions/SelfPower.cc
  transformations/functions/Exp.cc
  transformations/hist/HistEdges.cc
  transformations/hist/Rebin.cc
  transformations/integrator/GSLSamplerGL.cc
  transformations/integrator/IntegratorBase.cc
  transformations/integrator/Integrator21Base.cc
  transformations/integrator/Integrator2Base.cc
  transformations/integrator/IntegratorGL.cc
  transformations/integrator/Integrator21GL.cc
  transformations/integrator/Integrator2GL.cc
  transformations/integrator/IntegratorRect.cc
  transformations/integrator/IntegratorTrap.cc
  transformations/interpolation/LinearInterpolator.cc
  transformations/interpolation/InSegment.cc
  transformations/interpolation/SegmentWise.cc
  transformations/interpolation/InterpExpo.cc
  transformations/interpolation/InterpExpoU.cc
  transformations/linalg/Cholesky.cc
  transformations/linalg/Normalize.cc
  transformations/linalg/RenormalizeDiag.cc
  transformations/linalg/MatrixProduct.cc
  transformations/linalg/Transpose.cc
  transformations/neutrino/EvisToEe.cc
  transformations/neutrino/IbdFirstOrder.cc
  transformations/neutrino/IbdInteraction.cc
  transformations/neutrino/IbdZeroOrder.cc
  transformations/neutrino/Neutrino.cc
  transformations/neutrino/OscillationVariables.cc
  transformations/neutrino/OscProb2nu.cc
  transformations/neutrino/OscProbMatterUniform.cc
  transformations/neutrino/OscProbPMNS.cc
  transformations/neutrino/OscProbPMNSDecoh.cc
  transformations/neutrino/PDGVariables.cc
  transformations/neutrino/PMNSVariables.cc
  transformations/neutrino/OscProbPMNSVariables.cc
  transformations/neutrino/ReactorGroup.cc
  transformations/neutrino/ReactorNorm.cc
  transformations/neutrino/Units.cc
  transformations/stats/Chi2.cc
  transformations/stats/CovarianceToyMC.cc
  transformations/stats/CovariatedPrediction.cc
  transformations/stats/Covmat.cc
  transformations/stats/NormalToyMC.cc
  transformations/stats/Poisson.cc
  transformations/stats/PoissonToyMC.cc
  transformations/stats/ParCovMatrix.cc
  transformations/var/VarDiff.cc
  transformations/var/VarProduct.cc
  transformations/var/VarSum.cc
  transformations/var/VarArray.cc
  transformations/var/ArraySum.cc
  transformations/backgrounds/GeoNeutrinoFluxNormed.cc
  transformations/legacy/EnergyResolutionC.cc
  transformations/legacy/GaussLegendre.cc
  transformations/legacy/GaussLegendre2d.cc
  extra/GridFilter.cc
  examples/GaussianPeakWithBackground.cc
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
  Storage.hh
  Args.hh
  Rets.hh
  Ints.hh
  Rtypes.hh
  Itypes.hh
  Atypes.hh
  TypesFunctions.hh
  EntryHandle.hh
  OutputHandle.hh
  OpenHandle.hh
  InputHandle.hh
  TransformationDebug.hh
  TransformationBase.hh
  TransformationFunction.hh
  TransformationFunctionArgs.hh
  FunctionDescriptor.hh
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
  Concat.hh
  CovariatedPrediction.hh
  TrialMultiFunc.hh
  Product.hh
  Sum.hh
  Ratio.hh
  WeightedSum.hh
  Bins.hh
  Points.hh
  Histogram.hh
  Histogram2d.hh
  HistEdges.hh
  Identity.hh
  Dummy.hh
  LinearInterpolator.hh
  InSegment.hh
  SegmentWise.hh
  InterpExpo.hh
  InterpExpoU.hh
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
  ArraySum.hh
  Minimizable.hh
  Jacobian.hh
  Neutrino.hh
  OscillationVariables.hh
  PMNSVariables.hh
  OscProbPMNSVariables.hh
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
  IntegratorBase.hh
  Integrator21Base.hh
  Integrator2Base.hh
  IntegratorGL.hh
  Integrator21GL.hh
  Integrator2GL.hh
  IntegratorRect.hh
  IntegratorTrap.hh
  GSLSamplerGL.hh
  GaussLegendre.hh
  GaussLegendre2d.hh
  EnergyResolution.hh
  EnergyResolutionC.hh
  HistNonlinearity.hh
  HistSmear.hh
  HistSmearSparse.hh
  RenormalizeDiag.hh
  C14Spectrum.hh
  # examples/
  GaussianPeakWithBackground.hh
  # extra/
  GridFilter.hh
  EigenHelpers.hh
  ParCovMatrix.hh
  MatrixProduct.hh
  Transpose.hh
  )

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
                    transformations/legacy/
                    extra/
                    examples/
                   )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Setting ${CMAKE_BUILD_TYPE} build")

option(GENERATE_PROFILE OFF)
option(USE_PROFILE OFF)


set(PGO_COMPILE_FLAG "")
set(PGO_LINK_FLAG "")

if(GENERATE_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate=${CMAKE_BINARY_DIR}/profile-data")
    message(STATUS "Building with collection of PGO profiling enabled")
endif()

if(USE_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use=${CMAKE_BINARY_DIR}/profile-data -fprofile-correction")
    message(STATUS "Build using collected PGO profiling info")
endif()


set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)
message(STATUS "Compiling in C++${CMAKE_CXX_STANDARD} mode")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Werror -Wall  -pedantic -pipe")

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebugInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ggdb3 -fsanitize=undefined -fno-omit-frame-pointer")
endif()

include(dev-tools)

add_subdirectory(fmt EXCLUDE_FROM_ALL)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "$ENV{ROOTSYS}/etc/cmake")

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(Boost 1.40.0 REQUIRED)
if(Boost_FOUND)
    message(STATUS "found Boost version ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION} in ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
    message(FATAL_ERROR "Boost >= 1.40.0 not found")
endif(Boost_FOUND)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(GSL REQUIRED)
find_package(ROOT REQUIRED)
link_directories(${ROOT_LIBRARY_DIR})

ROOT_GENERATE_DICTIONARY(Dict "${HEADERS}" OPTIONS -interpreteronly)

add_library(${LIBNAME} SHARED ${SOURCES} Dict.cxx ${CONTRIB_SRC})
target_include_directories(${LIBNAME} SYSTEM PRIVATE ${ROOT_INCLUDE_DIR})
target_include_directories(${LIBNAME} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(${LIBNAME} PRIVATE  ${ROOT_LIBRARIES} -lPyROOT)
target_link_libraries(${LIBNAME} PRIVATE GSL::gsl GSL::gslcblas)
target_link_libraries(${LIBNAME} PRIVATE fmt::fmt-header-only)

if(GENERATE_PROFILE)
    target_link_libraries(${LIBNAME} PRIVATE "-fprofile-generate=${CMAKE_BINARY_DIR}/profile-data")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebugInfo")
    target_link_libraries(${LIBNAME} PRIVATE "-fsanitize=undefined")
endif()
