#include "GPUVariables.hh"
#include "ParametrizedBase.hh"

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::readVariables(ParametrizedTypes::ParametrizedBase* parbase){
	for(auto& entry: parbase->m_entries){
		auto& var=entry.var;
		if(!var.istype<FloatType>()){
			throw std::runtime_error("GPUVariablesLocal: Invalid variable type");
		}
		m_variables.emplace_back(variable<FloatType>(var));
	}

	//setSize(vars.size());
	//size_t i(0);
	//for (auto& var: vars) {
		//readVariable(i, var);
		//++i;
	//}
	//syncHost2Device();
}

template class TransformationTypes::GPUVariables<double>;

#ifdef PROVIDE_SINGLE_PRECISION
	template class TransformationTypes::GPUVariables<float>;
#endif
