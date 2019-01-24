#include "GPUFunctionData.hh"
#include "GPUFunctionArgs.hh"

template class TransformationTypes::GPUFunctionData<double>;

#ifdef PROVIDE_SINGLE_PRECISION
	template class TransformationTypes::GPUFunctionData<float>;
#endif
