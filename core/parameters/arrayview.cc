#include "arrayview.hh"

/* template struct std::complex<double>; */
template class arrayview<double>;
#ifdef PROVIDE_SINGLE_PRECISION
	/* template struct std::complex<float>; */
	template class arrayview<float>;
#endif
