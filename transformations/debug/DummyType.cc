#include "DummyType.hh"

using TransformationTypes::FunctionArgsT;
using GNA::GNAObjectTemplates::DummyTypeT;

template class GNA::GNAObjectTemplates::DummyTypeT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::DummyTypeT<float>;
#endif
