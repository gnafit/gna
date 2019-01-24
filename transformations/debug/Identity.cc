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
		.func("identity_gpuargs", identity_gpu/*, DataLocation::Device*/)
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

using TransformationTypes::GPUShape;

void identity_gpu(FunctionArgs& fargs){
	fargs.args.touch();

	auto* source=fargs.gpu->args[0];
	auto* dest  =fargs.gpu->rets[0];
	auto* shape =fargs.gpu->argshapes[0];
	auto bytes=shape[(int)GPUShape::Size]*sizeof(decltype(source[0]));
	//printf("copy %p->%p size %zu\n", (void*)source, (void*)dest, fargs.gpu->argshapes[0][(int)GPUShape::Size]);
	memcpy(dest, source, bytes);
	fargs.gpu->dump();
}
