#include "Identity.hh"

#include <iostream>

#include "TypeClasses.hh"
using namespace TypeClasses;

#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#endif

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    IdentityT<FloatType>::IdentityT(){
        this->transformation_("identity")
            .input("source")
            .output("target")
            .types(new CheckSameTypesT<FloatType>({0,-1}), new PassTypeT<FloatType>(0, {0,0}))
            .func([](typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs){ fargs.rets[0].x = fargs.args[0].x; })
    #ifdef GNA_CUDA_SUPPORT     //
            .func("identity_gpuargs_h", &IdentityT<FloatType>::identity_gpu_h, DataLocation::Host)
            .func("identity_gpuargs_d", &IdentityT<FloatType>::identity_gpu_d, DataLocation::Device)
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

      #ifdef GNA_CUDA_SUPPORT
      template<typename FloatType>
      void IdentityT<FloatType>::identity_gpu_h(typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs){
          fargs.args.touch();
          auto& gpuargs=fargs.gpu;
          gpuargs->provideSignatureHost(/*local*/true);

          auto* source=*gpuargs->args;
          auto* dest  =*gpuargs->rets;
          auto* shape =*gpuargs->argshapes;
          auto bytes=shape[(int)TransformationTypes::GPUShape::Size]*sizeof(decltype(source[0]));
          //printf("copy %p->%p size %zu\n", (void*)source, (void*)dest, fargs.gpu->argshapes[0][(int)GPUShape::Size]);
          memcpy(dest, source, bytes);
          fargs.gpu->dump();
      }

      template<typename FloatType>
      void IdentityT<FloatType>::identity_gpu_d(typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs){
          fargs.args.touch();
          auto& gpuargs=fargs.gpu;
      //  gpuargs->provideSignatureDevice();
      //        gpuargs->setAsDevice();
          auto** source=gpuargs->args;
          auto** dest  =gpuargs->rets;
          identity_gpu(source, dest, gpuargs->nargs, fargs.args[0].arr.size());
          fargs.args[0].gpuArr->dump();
          fargs.rets[0].gpuArr->dump();
          //gpuargs->setAsDevice();
      }
      #endif
    }
  }



template class GNA::GNAObjectTemplates::IdentityT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::IdentityT<float>;
#endif
