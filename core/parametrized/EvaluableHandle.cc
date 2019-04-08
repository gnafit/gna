#include "EvaluableHandle.hh"

template class ParametrizedTypes::EvaluableHandle<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class ParametrizedTypes::EvaluableHandle<float>;
#endif
