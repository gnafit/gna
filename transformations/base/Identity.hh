#pragma once

#include <iostream>
#include "GNAObject.hh"
#include "TypesFunctions.hh"

//
// Identity transformation
//
class Identity: public GNASingleObject,
                public TransformationBind<Identity> {
public:
  bool isgpu = false;
  Identity(bool gpu = false) : isgpu(gpu) {
    transformation_("identity")
      .input("source")
      .output("target")
      .types(TypesFunctions::pass<0,0>)
      .func(&Identity::ident)
#ifdef GNA_CUDA_SUPPORT
      .setEntryLocation(gpu? DataLocation::Device : DataLocation::Host)
#endif
      ;
  };

  void ident(Args args, Rets rets) {
#ifdef GNA_CUDA_SUPPORT
    if (isgpu) gpu_test(args, rets);
    else 
#endif
      rets[0].x = 2*args[0].x;
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

