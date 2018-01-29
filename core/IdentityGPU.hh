#ifdef GNA_CUDA_SUPPORT

#ifndef IDENTITYGPU_H
#define IDENTITYGPU_H 1

#include <stdio.h>
#include "extra/GNAcuGpuArray.hh"

//
// IdentityGPU transformation
//
class IdentityGPU: public GNAObject,
                public Transformation<IdentityGPU> {
public:
  IdentityGPU(){
    transformation_(this, "identitygpu")
      .setEntryLocation(Device)
      .input("source")
      .output("target")
      .types(Atypes::pass<0>)
      .func([](Args args, Rets rets){ 
		rets[0].gpuArr->setByDeviceArray(args[0].gpuArr->devicePtr); 
		*(rets[0].gpuArr) *=15;
		std::cout << "Dump: ";
		rets[0].gpuArr->dump();
		})
      ;
  };

  void dump(){
      auto& data = t_["identitygpu"][0];

      if( data.type.shape.size()==2u ){
          std::cout<<data.arr2d<<std::endl;
      }
      else{
          std::cout<<data.arr<<std::endl;
      }
  }
};

#endif
#endif
