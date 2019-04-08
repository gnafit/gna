#include "VariableHandle.hh"

template class ParametrizedTypes::VariableHandle<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class ParametrizedTypes::VariableHandle<float>;
#endif
