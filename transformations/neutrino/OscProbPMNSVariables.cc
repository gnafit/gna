#include "OscProbPMNSVariables.hh"

template class GNA::GNAObjectTemplates::OscProbPMNSExpressionsT<double>;
template class GNA::GNAObjectTemplates::OscProbPMNSVariablesT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::OscProbPMNSExpressionsT<float>;
  template class GNA::GNAObjectTemplates::OscProbPMNSVariablesT<float>;
#endif
