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

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::setSize(size_t size){
	if(names.size()==size){
		return;
	}
	deAllocateDevice();

	names.resize(size);
	h_values.resize(size);
	h_value_pointers_host.resize(size);
	h_value_pointers_dev.resize(size);

	auto* ptr=h_values.data();
	for (size_t i = 0; i < h_value_pointers_host.size(); ++i) {
		h_value_pointers_host[i]=ptr;
		std::advance(ptr, 1);
	}

	allocateDevice();
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::deAllocateDevice(){
#ifdef GNA_CUDA_SUPPORT
	if(d_values){
		cuwr_free<FloatType>(d_values);
		d_values=nullptr;
	}
	if(d_value_pointers_dev){
		/// TODO if I need to free each cont
		cuwr_free<FloatType*>(d_value_pointers_dev);
		d_value_pointers_dev=nullptr;
	}
#endif
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::allocateDevice(){
#ifdef GNA_CUDA_SUPPORT
	/// allocate d_values (same as h_values, no sync is needed here)
	device_malloc(d_values, h_values.size());
	auto* ptr=d_values;
	for (size_t i = 0; i < h_value_pointers_dev.size(); ++i) {
		h_value_pointers_dev[i]=ptr;
		ptr = ptr + 1;
	}
	copyH2D_ALL(d_value_pointers_dev, h_value_pointers_dev.data(), h_value_pointers_dev.size());
#endif
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::dump(const std::string& type){
	size_t nvars=h_values.size();

	printf("Dumping GPUVariablesLocal of size %zu", nvars);
	if(type.size()){
		printf(" (%s)", type.c_str());
	}
	printf("\n");
	for (size_t i = 0; i < nvars; ++i) {
		auto  value=*h_value_pointers_host[i];
		auto& name=names[i];

		printf("  Variable %zu: %12.6g    %s\n", i, value, name.c_str());
	}
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::provideSignatureHost(SizeType &nvars, FloatType** &values){
	nvars=h_value_pointers_host.size();
	values=h_value_pointers_host.data();
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::provideSignatureDevice(SizeType &nvars, FloatType** &values){
	nvars=h_value_pointers_dev.size();
	values=d_value_pointers_dev;
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::readVariables(){
	setSize(m_variables.size());
	size_t i(0);
	for (auto& var: m_variables) {
		readVariable(i, var);
		++i;
	}
	syncHost2Device();
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::readVariable(size_t i, const variable<FloatType>& var){
	names[i]=var.name();
	h_values[i]=var.value();
}

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariablesLocal<FloatType,SizeType>::syncHost2Device(){
#ifdef GNA_CUDA_SUPPORT
	/// h_values -> d_values
	copyH2D_NA(d_values, h_values.data(), h_values.size());
#endif
}

template class TransformationTypes::GPUVariablesLocal<double,size_t>;
#ifdef PROVIDE_SINGLE_PRECISION
	template class TransformationTypes::GPUVariablesLocal<float,size_t>;
#endif
