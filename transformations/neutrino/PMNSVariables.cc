#include "PMNSVariables.hh"

template class GNA::GNAObjectTemplates::PMNSExpressionsT<double>;
template class GNA::GNAObjectTemplates::PMNSVariablesT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::PMNSExpressionsT<float>;
  template class GNA::GNAObjectTemplates::PMNSVariablesT<float>;
#endif
