#include "OscillationVariables.hh"

template class GNA::GNAObjectTemplates::OscillationExpressionsT<double>;
template class GNA::GNAObjectTemplates::OscillationVariablesT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::OscillationExpressionsT<float>;
  template class GNA::GNAObjectTemplates::OscillationVariablesT<float>;
#endif
