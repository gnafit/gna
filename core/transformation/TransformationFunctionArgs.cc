#include "TransformationFunctionArgs.hh"

void TransformationTypes::FunctionArgs::requireGPU(){
	if(gpu){
		return;
	}

	gpu.reset(new GPUFunctionArgs(m_entry));
}

void TransformationTypes::FunctionArgs::updateTypes(){
	if(gpu){
		gpu->updateTypes();
	}
}

