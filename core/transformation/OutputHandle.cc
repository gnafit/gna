#include "OutputHandle.hh"

template class TransformationTypes::OutputHandleT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::OutputHandleT<float>;
#endif

