#include "TreeManager.hh"

template class GNA::TreeManager<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::TreeManager<float>;
#endif
