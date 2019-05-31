#include "Exp.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>
#include "config_vars.h"
#include "cuElementary.hh"


/**
 * @brief Constructor.
 */
Exp::Exp() {
    transformation_("exp")
        .input("points")
	.output("result")
	.types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
	.func(&Exp::calculate)
#ifdef GNA_CUDA_SUPPORT
	.func("gpu", &Exp::calc_gpu, DataLocation::Device)
#endif
      ;
}

/**
 * @brief Calculate the value of function.
 */
void Exp::calculate(FunctionArgs& fargs){
    fargs.rets[0].x = fargs.args[0].x.exp();
}


void Exp::calc_gpu(FunctionArgs& fargs) {
        fargs.args.touch();
        auto& gpuargs=fargs.gpu;
        gpuargs->provideSignatureDevice();
        auto** source=gpuargs->args;
        auto** dest  =gpuargs->rets;
        cuexp(source, dest, gpuargs->nargs, fargs.args[0].arr.size());
}
