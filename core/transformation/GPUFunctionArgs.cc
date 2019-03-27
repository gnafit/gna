#include "GPUFunctionArgs.hh"

template class TransformationTypes::GPUFunctionArgsT<double,size_t>;
#ifdef PROVIDE_SINGLE_PRECISION
template class TransformationTypes::GPUFunctionArgsT<float,size_t>;
#endif
