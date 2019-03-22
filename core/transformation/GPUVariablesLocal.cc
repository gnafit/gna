#include "GPUVariablesLocal.hh"
#include "ParametrizedBase.hh"

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::readVariables(ParametrizedTypes::ParametrizedBase* parbase){
	for(auto& entry: parbase->m_entries){
		auto& var=entry.var;
		if(!var.istype<FloatType>()){
			throw std::runtime_error("GPUVariablesLocal: Invalid variable type");
		}
		m_variables.emplace_back(variable<FloatType>(var));
	}
}

template class TransformationTypes::GPUVariablesLocal<double>;

#ifdef PROVIDE_SINGLE_PRECISION
	template class TransformationTypes::GPUVariablesLocal<float>;
#endif
