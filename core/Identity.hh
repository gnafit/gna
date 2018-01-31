#ifndef IDENTITY_H
#define IDENTITY_H 1

#include <stdio.h>
#ifdef GNA_CUDA_SUPPORT
#include "extra/GNAcuGpuArray.hh"
#endif
//
// Identity transformation
//i
class Identity: public GNASingleObject,
                public Transformation<Identity> {
public:
  Identity(bool is_gpu = false) : isgpu(is_gpu) {
    transformation_(this, "identity")
      .setEntryLocation(
#ifdef GNA_CUDA_SUPPORT
              is_gpu ? Device : Host
#endif
              )
      .input("source")
      .output("target")
      .types(Atypes::pass<0,0>)
      .func(&Identity::identity)
      ;
  };

  bool isgpu = false;

  void identity (Args args, Rets rets) {
    if (!isgpu) rets[0].x = args[0].x;
#ifdef GNA_CUDA_SUPPORT
    else { gpu_test(args, rets); }
#endif
  }
#ifdef GNA_CUDA_SUPPORT
  void gpu_test (Args args, Rets rets) {
    rets[0].gpuArr->setByDeviceArray(args[0].gpuArr->devicePtr);
    *(rets[0].gpuArr) *=15;
    std::cout << "Dump: ";
    rets[0].gpuArr->dump();
  }
#endif
  void dump(){
      auto& data = t_["identity"][0];

      if( data.type.shape.size()==2u ){
          std::cout<<data.arr2d<<std::endl;
      }
      else{
          std::cout<<data.arr<<std::endl;
      }
  }
};

#endif
