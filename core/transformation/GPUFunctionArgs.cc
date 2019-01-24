#include "GPUFunctionArgs.hh"

template class TransformationTypes::GPUFunctionArgsT<double>;

#ifdef PROVIDE_SINGLE_PRECISION
	template class TransformationTypes::GPUFunctionArgsT<float>;
#endif
