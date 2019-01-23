#include "Identity.hh"

using TransformationTypes::FunctionArgs;

void identity_gpu(FunctionArgs& fargs);

Identity::Identity(){
	transformation_("identity")
		.input("source")
		.output("target")
		.types(TypesFunctions::pass<0,0>)
		.func([](FunctionArgs& fargs){ fargs.rets[0].x = fargs.args[0].x; })
//#ifdef GNA_CUDA_SUPPORT     //
		.func("identity_gpu", identity_gpu/*, DataLocation::Device*/)
//#endif
		;
}

void Identity::dump(){
	auto& data = t_["identity"][0];

	if( data.type.shape.size()==2u ){
		std::cout<<data.arr2d<<std::endl;
	}
	else{
		std::cout<<data.arr<<std::endl;
	}
}

void identity_gpu(FunctionArgs& fargs){
	fargs.gpu->dump();

	memcpy(fargs.gpu->rets[0], fargs.gpu->args[0], fargs.gpu->argshapes[0][1]);
}
