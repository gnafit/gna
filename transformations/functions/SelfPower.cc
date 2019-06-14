#include "SelfPower.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>
#include <cmath>

#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#include "GpuBasics.hh"
#include "DataLocation.hh"
#endif

/**
 * @brief Constructor.
 *
 * Creates two transformations for functions with positive and negative power.
 *
 * @param scalename="sp_scale" -- name of a variable to scale the argument.
 */
SelfPower::SelfPower(const char* scalename/*="sp_scale"*/) {
    variable_(&m_scale, scalename);

    transformation_("selfpower")
        .input("points")
	.output("result")
	.types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
	.func(&SelfPower::calculate)
#ifdef GNA_CUDA_SUPPORT
	.func("gpu", &SelfPower::gpu_calculate, DataLocation::Device)
#endif
      ;

    transformation_("selfpower_inv")
        .input("points")
	.output("result")
	.types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
	.func(&SelfPower::calculate_inv)
      ;
}

void SelfPower::gpu_calculate(FunctionArgs& fargs) {
    fargs.args.touch();
    auto& gpuargs=fargs.gpu;
    gpuargs->provideSignatureDevice();
    cuselfpower(gpuargs->args, gpuargs->rets, fargs.args[0].arr.size(), gpuargs->nargs,m_scale.value());
}

/**
 * @brief Calculate the value of function with positive power.
 */
void SelfPower::calculate(FunctionArgs& fargs){
    auto& res = fargs.rets[0].x = fargs.args[0].x/m_scale.value();
    res=res.pow(res);
}

/**
 * @brief Calculate the value of function with negative power.
 */
void SelfPower::calculate_inv(FunctionArgs& fargs){
    auto& res = fargs.rets[0].x = fargs.args[0].x/m_scale.value();
    res=res.pow(-res);
}
