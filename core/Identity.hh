#ifndef IDENTITY_H
#define IDENTITY_H 1

#include <stdio.h>
#include "extra/GNAcuGpuArray.hh"

//
// Identity transformation
//
class Identity: public GNASingleObject,
                public Transformation<Identity> {
public:
  Identity(){
    transformation_(this, "identity")
      .input("source")
      .output("target")
      .types(Atypes::pass<0>)
      .func(&Identity::identity)
      ;
    auto gpu_test = transformation_(this, "gpu_test")
      .setEntryLocation(Device)
      .input("source")
      .output("target")
      .types(Atypes::pass<0>)
      .func(&Identity::gpu_test)
      ;

  }

  void identity (Args args, Rets rets) {
    rets[0].x = args[0].x;
  }

  void gpu_test (Args args, Rets rets) {
    rets[0].gpuArr->setByDeviceArray(args[0].gpuArr->devicePtr);
    *(rets[0].gpuArr) *=15;
    std::cout << "Dump: ";
    rets[0].gpuArr->dump();
  }

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
