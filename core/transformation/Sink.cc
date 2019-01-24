#include "Sink.hh"

template class TransformationTypes::SinkT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::SinkT<float>;
#endif

