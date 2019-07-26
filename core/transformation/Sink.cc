#include "Sink.hh"

template struct TransformationTypes::SinkT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template struct TransformationTypes::SinkT<float>;
#endif

