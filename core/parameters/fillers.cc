#include "fillers.hh"

template size_t copyVariableToArray<>(const variable<double>& var, double* dest);
template size_t copyArrayToParameter<>(double* source, parameter<double>& par);
template size_t copyVariableToArray<>(const variable<std::array<double,2>>& var, double* dest);
template size_t copyVariableToArray<>(const variable<std::array<double,3>>& var, double* dest);
template size_t copyArrayToParameter<>(double* source, parameter<std::array<double,2>>& par);
template size_t copyArrayToParameter<>(double* source, parameter<std::array<double,3>>& par);

#ifdef PROVIDE_SINGLE_PRECISION
	template size_t copyVariableToArray<>(const variable<float>& var, float* dest);
	template size_t copyArrayToParameter<>(float* source, parameter<float>& par);
	template size_t copyVariableToArray<>(const variable<std::array<float,2>>& var, float* dest);
	template size_t copyVariableToArray<>(const variable<std::array<float,3>>& var, float* dest);
	template size_t copyArrayToParameter<>(float* source, parameter<std::array<float,2>>& par);
	template size_t copyArrayToParameter<>(float* source, parameter<std::array<float,3>>& par);
#endif

