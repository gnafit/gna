set(CORE_SOURCES
  # Common classes
  core/TreeManager.cc
  # Parameters
  core/parameters/arrayview.cc
  core/parameters/changeable.cc
  core/parameters/variable.cc
  core/parameters/parameter.cc
  core/parameters/evaluable.cc
  core/parameters/dependant.cc
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
  core/transformation/OpenHandle.cc
  core/transformation/Initializer.cc
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
  core/transformation/TypeClasses.cc
  core/parameters/fillers.cc
  # GNAObject and bindings
  core/TransformationDescriptor.cc
  core/GNAObject.cc
  core/GNAObjectBind1N.cc
  core/GNAObjectBindMN.cc
  # core/parametrized
  core/parametrized/ParametrizedEntry.cc
  core/parametrized/ParametrizedBase.cc
  core/parametrized/VariableHandle.cc
  core/parametrized/EvaluableEntry.cc
  core/parametrized/EvaluableHandle.cc
  core/parametrized/VariableDescriptor.cc
  core/parametrized/EvaluableDescriptor.cc
  # Parameter representation
  core/ParametersGroup.cc
  core/UncertainParameter.cc
  PARENT_SCOPE
 )

set(CORE_HEADERS
  # core/ Common classes
  TreeManager.hh
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
  TypeClasses.hh
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
  TransformationBind.hh
  SingleOutput.hh
  Initializer.hh
  Accessor.hh
  # variables
  arrayview.hh
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
  # core/parametrized
  ParametrizedEntry.hh
  ParametrizedBase.hh
  VariableHandle.hh
  EvaluableEntry.hh
  EvaluableHandle.hh
  VariableDescriptor.hh
  EvaluableDescriptor.hh
  # Parameter representation
  UncertainParameter.hh
  # core/
  TransformationDescriptor.hh
  GNAObject.hh
  GNAObjectBind1N.hh
  GNAObjectBindMN.hh
  PARENT_SCOPE
 )
