#include "GPUVariablesLocal.hh"

template class TransformationTypes::GPUVariablesLocal<double>;

#ifdef PROVIDE_SINGLE_PRECISION
	template class TransformationTypes::GPUVariablesLocal<float>;
#endif
