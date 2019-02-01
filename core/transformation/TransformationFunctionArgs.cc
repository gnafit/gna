#include "TransformationFunctionArgs.hh"
#include "GPUFunctionArgs.hh"

template<typename SourceFloatType, typename SinkFloatType>
TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>::~FunctionArgsT<SourceFloatType,SinkFloatType>(){

}

template<typename SourceFloatType, typename SinkFloatType>
void TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>::requireGPU(){
	if(gpu){
		return;
	}

	gpu.reset(new GPUFunctionArgsType(m_entry));
}

template<typename SourceFloatType, typename SinkFloatType>
void TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>::updateTypes(){
	if(gpu){
		gpu->updateTypes();
	}
}

template class TransformationTypes::FunctionArgsT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::FunctionArgsT<float,float>;
#endif
