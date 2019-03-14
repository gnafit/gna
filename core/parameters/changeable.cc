#include "changeable.hh"

template class inconstant_data<double>;
#ifdef PROVIDE_SINGLE_PRECISION
	template class inconstant_data<float>;
#endif
