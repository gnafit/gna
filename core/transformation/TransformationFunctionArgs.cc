#include "TransformationFunctionArgs.hh"
#include "GPUFunctionArgs.hh"
#include "ParametrizedBase.hh"

template<typename SourceFloatType, typename SinkFloatType>
TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>::~FunctionArgsT<SourceFloatType,SinkFloatType>(){

}

template<typename SourceFloatType, typename SinkFloatType>
void TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>::readVariables(ParametrizedTypes::ParametrizedBase* parbase){
	if(parbase==m_parbase){
		return;
	}
	m_parbase = parbase;

	this->readVariables();
}

template<typename SourceFloatType, typename SinkFloatType>
void TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>::readVariables(){
	if(!m_parbase){
		return;
		//throw std::runtime_error("No ParametrizedBase pointer is set. Should not happen.");
	}

  if(this->gpu){
    this->gpu->readVariables(m_parbase);
  }
}

template<typename SourceFloatType, typename SinkFloatType>
void TransformationTypes::FunctionArgsT<SourceFloatType,SinkFloatType>::requireGPU(){
	if(gpu){
		return;
	}

	gpu.reset(new GPUFunctionArgsType(m_entry));
	if(m_parbase){
    this->gpu->readVariables(m_parbase);
	}
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
