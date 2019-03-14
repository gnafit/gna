#include "arrayview.hh"

template class std::complex<double>;
template class arrayview<double>;
#ifdef PROVIDE_SINGLE_PRECISION
	template class std::complex<float>;
	template class arrayview<float>;
#endif
