#include "dependant.hh"

template class dependant<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class dependant<float>;
#endif
