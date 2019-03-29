#include "arrayviewAllocator.hh"

template class arrayviewAllocatorSimple<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class arrayviewAllocatorSimple<float>;
#endif
