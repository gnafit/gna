#include "parameter.hh"

template class parameter<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class parameter<float>;
#endif
