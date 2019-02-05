#include "Identity.hh"

#include <iostream>

#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#endif

using FunctionArgs = TransformationTypes::FunctionArgsT<double,double>;

void identity_gpu_h(FunctionArgs& fargs);
void identity_gpu_d(FunctionArgs& fargs);

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    IdentityT<FloatType>::IdentityT(){
    	this->transformation_("identity")
    		.input("source")
    		.output("target")
    		.types(TypesFunctions::ifSame,TypesFunctions::pass<0,0>)
    		.func([](FunctionArgs& fargs){ fargs.rets[0].x = fargs.args[0].x; })
    		.func("identity_gpuargs_h", identity_gpu_h, DataLocation::Host)
    #ifdef GNA_CUDA_SUPPORT     //
    		.func("identity_gpuargs_d", identity_gpu_d, DataLocation::Device)
    #endif
    		;
    }
    
    template<typename FloatType>
    void IdentityT<FloatType>::dump(){
    	auto& data = this->t_["identity"][0];
    
    	if( data.type.shape.size()==2u ){
    		std::cout<<data.arr2d<<std::endl;
    	}
    	else{
    		std::cout<<data.arr<<std::endl;
    	}
    }
  }
}

using TransformationTypes::GPUShape;

void identity_gpu_h(FunctionArgs& fargs){
	fargs.args.touch();
	auto& gpuargs=fargs.gpu;
	gpuargs->provideSignatureHost();

	auto* source=*gpuargs->args;
	auto* dest  =*gpuargs->rets;
	auto* shape =*gpuargs->argshapes;
	auto bytes=shape[(int)GPUShape::Size]*sizeof(decltype(source[0]));
	//printf("copy %p->%p size %zu\n", (void*)source, (void*)dest, fargs.gpu->argshapes[0][(int)GPUShape::Size]);
	memcpy(dest, source, bytes);
	fargs.gpu->dump();
}

void identity_gpu_d(FunctionArgs& fargs){
	fargs.args.touch();
	auto& gpuargs=fargs.gpu;
//	gpuargs->provideSignatureDevice();

	auto** source=gpuargs->args;
	auto** dest  =gpuargs->rets;
	identity_gpu(source, dest, gpuargs->nargs, fargs.args[0].arr.size());
	fargs.args[0].gpuArr->dump();
	fargs.rets[0].gpuArr->dump();
}
template class GNA::GNAObjectTemplates::IdentityT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::IdentityT<float>;
#endif
