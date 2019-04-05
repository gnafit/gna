#include <boost/math/constants/constants.hpp>

#include <stdexcept>

#include "UncertainParameter.hh"

template class GaussianParameter<double>;
template class Variable<double>;
template class Parameter<double>;
template class ParameterWrapper<double>;
template class UniformAngleParameter<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GaussianParameter<float>;
  template class Variable<float>;
  template class Parameter<float>;
  template class ParameterWrapper<float>;
  template class UniformAngleParameter<float>;
#endif
