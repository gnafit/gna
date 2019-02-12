#include "OpenHandle.hh"

template class OpenHandleT<double,double>;
template class OpenOutputHandleT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class OpenHandleT<float,float>;
  template class OpenOutputHandleT<float,float>;
#endif
