#include "variable.hh"

template class variable<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class variable<float>;
#endif
