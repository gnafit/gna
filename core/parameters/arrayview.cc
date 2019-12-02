#include "arrayview.hh"

template class arrayview<double>;
#ifdef PROVIDE_SINGLE_PRECISION
	template class arrayview<float>;
#endif
