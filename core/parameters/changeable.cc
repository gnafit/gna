#include "changeable.hh"

template struct inconstant_data<double>;
#ifdef PROVIDE_SINGLE_PRECISION
	template struct inconstant_data<float>;
#endif
