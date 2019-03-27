#include "GPUVariables.hh"
#include "ParametrizedBase.hh"
#include "TreeManager.hh"
#include "TransformationEntry.hh"

using GNA::TreeManager;

template<typename FloatType,typename SizeType>
TransformationTypes::GPUVariables<FloatType,SizeType>::GPUVariables(TransformationTypes::EntryT<FloatType,FloatType>* transformation){
	m_tmanager = TreeManager<FloatType>::current();
	if(!m_tmanager){
		return;
	}

	if(m_tmanager!=transformation->m_tmanager){
		throw std::runtime_error("Transformation is not managed by the current TreeManager");
	}
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::readVariables(ParametrizedTypes::ParametrizedBase* parbase){
	if(!m_tmanager){
		return;
	}
	for(auto& entry: parbase->m_entries){
		auto& var=entry.var;
		if(!var.istype<FloatType>()){
			throw std::runtime_error("GPUVariablesLocal: Invalid variable type");
		}
		if(!m_tmanager->hasVariable(var)){
			throw std::runtime_error("The variable is not known to the current TreeManager");
		}
		m_variables.emplace_back(variable<FloatType>(var));
	}

	auto size=m_variables.size();
	h_value_pointers_host.resize(size);
	h_value_pointers_dev.resize(size);

	deAllocateDevice();

	FloatType* m_dev_root=nullptr;
	for (size_t i = 0; i < size; ++i) {
		auto& val=m_variables[i].values();
		h_value_pointers_host[i]=val.data();
		h_value_pointers_dev[i]= m_dev_root+val.offset();
	}
	allocateDevice();
}

template class TransformationTypes::GPUVariables<double,size_t>;
#ifdef PROVIDE_SINGLE_PRECISION
template class TransformationTypes::GPUVariables<float,size_t>;
#endif
