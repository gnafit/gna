#include "Exp.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>
#include "config_vars.h"

#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#endif

namespace GNA {
  namespace GNAObjectTemplates {
    /**
     * @brief Constructor.
     */
    template<typename FloatType>
    ExpT<FloatType>::ExpT() {
        this->transformation_("exp")
            .input("points")
    	    .output("result")
    	    .types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
    	    .func(&ExpT<FloatType>::calculate)
    #ifdef GNA_CUDA_SUPPORT
    	    .func("gpu", &ExpT<FloatType>::calc_gpu, DataLocation::Device)
    #endif
          ;
    }
    
    /**
     * @brief Calculate the value of function.
     */
    template<typename FloatType>
    void ExpT<FloatType>::calculate(FunctionArgs& fargs){
        fargs.rets[0].x = fargs.args[0].x.exp();
    }
    
    
    template<typename FloatType>
    void ExpT<FloatType>::calc_gpu(FunctionArgs& fargs) {
            fargs.args.touch();
            auto& gpuargs=fargs.gpu;
            gpuargs->provideSignatureDevice();
            auto** source=gpuargs->args;
            auto** dest  =gpuargs->rets;
            cuexp(source, dest, gpuargs->nargs, fargs.args[0].arr.size());
    }
  }
}

template class GNA::GNAObjectTemplates::ExpT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::ExpT<float>;
#endif
