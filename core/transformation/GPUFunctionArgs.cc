#include "GPUFunctionArgs.hh"


template<typename FloatType,typename SizeType>
void TransformationTypes::GPUFunctionArgsT<FloatType,SizeType>::updateTypesHost(){
	m_args.fillContainers(m_entry->sources);
	m_rets.fillContainers(m_entry->sinks);
	m_ints.fillContainers(m_entry->storages);

	//provideSignatureHost();
}

#ifdef GNA_CUDA_SUPPORT
template<typename FloatType,typename SizeType>
void TransformationTypes::GPUFunctionArgsT<FloatType,SizeType>::updateTypesDevice(){
	m_args.fillContainersDevice(m_entry->sources);
	m_rets.fillContainersDevice(m_entry->sinks);
	m_ints.fillContainersDevice(m_entry->storages);
//	provideSignatureDevice();
//#else
	//std::cerr << "There is no CUDA support, so I can't switch your function to GPU-based one." << std::endl;
}
#endif

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUFunctionArgsT<FloatType,SizeType>::provideSignatureHost(bool local){
	if(local){
		m_vars.provideSignatureHost(nvars, vars);
	}
	else{
		m_vars_global.provideSignatureHost(nvars, vars);
	}
	m_args.provideSignatureHost(nargs, args, argshapes);
	m_rets.provideSignatureHost(nrets, rets, retshapes);
	m_ints.provideSignatureHost(nints, ints, intshapes);

	argsmapping = m_entry->mapping.size() ? m_entry->mapping.data() : nullptr;
}

#ifdef GNA_CUDA_SUPPORT
template<typename FloatType,typename SizeType>
void TransformationTypes::GPUFunctionArgsT<FloatType,SizeType>::provideSignatureDevice(bool local){
	if(local){
		m_vars.provideSignatureDevice(nvars, vars);
	}
	else{
		m_vars_global.provideSignatureDevice(nvars, vars);
	}
	m_args.provideSignatureDevice(nargs, args, argshapes);
	m_rets.provideSignatureDevice(nrets, rets, retshapes);
	m_ints.provideSignatureDevice(nints, ints, intshapes);

	argsmapping = m_argsmapping_dev;
}
#endif

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUFunctionArgsT<FloatType,SizeType>::dump(){
	printf("Dumping GPU args state\n");

	m_vars.dump("variables");
	printf("\n");

	m_vars_global.dump("variables");
	printf("\n");

	m_args.dump("sources");
	printf("\n");

	m_rets.dump("sinks");
	printf("\n");

	m_ints.dump("storages");
	printf("\n");
}

template class TransformationTypes::GPUFunctionArgsT<double,size_t>;
#ifdef PROVIDE_SINGLE_PRECISION
template class TransformationTypes::GPUFunctionArgsT<float,size_t>;
#endif
