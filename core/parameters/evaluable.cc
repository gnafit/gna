#include "evaluable.hh"

template class evaluable<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class evaluable<float>;
#endif
