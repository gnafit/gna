#include "GPUVariables.hh"
#include "GNAObject.hh"

template<typename FloatType,typename SizeType>
void TransformationTypes::GPUVariables<FloatType,SizeType>::readVariables(GNAObjectT<FloatType,FloatType>* object){
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
