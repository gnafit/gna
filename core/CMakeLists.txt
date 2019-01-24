set(CORE_SOURCES
  # GPU staff
  core/transformation/GPUFunctionArgs.cc
  core/transformation/GPUVariablesLocal.cc
  core/transformation/GPUFunctionData.cc
  # Transformation
  core/transformation/TransformationEntry.cc
  core/transformation/TransformationFunctionArgs.cc
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
  core/transformation/Sink.cc
  core/transformation/Storage.cc
  core/transformation/TransformationBase.cc
  core/transformation/TransformationErrors.cc
  core/transformation/TypesFunctions.cc
  core/parameters/fillers.cc
  core/GNAObject.cc
  core/GNAObjectBind1N.cc
  core/ParametersGroup.cc
  core/Parametrized.cc
  core/TransformationDescriptor.cc
  core/UncertainParameter.cc
  PARENT_SCOPE
 )

set(CORE_HEADERS
  # core/
  Data.hh
  Random.hh
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
  # variables
  changeable.hh
  variable.hh
  parameter.hh
  taintflag.hh
  fillers.hh
  dependant.hh
  evaluable.hh
  callback.hh
  # GPU related stuff
  GPUVariablesLocal.hh
  GPUFunctionData.hh
  GPUFunctionArgs.hh
  # core/
  GNAObject.hh
  GNAObjectBind1N.hh
  UncertainParameter.hh
  PARENT_SCOPE
 )
