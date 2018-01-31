#ifndef IDENTITY_H
#define IDENTITY_H 1

#include <stdio.h>
#include "extra/GNAcuGpuArray.hh"

//
// Identity transformation
//i
class Identity: public GNASingleObject,
                public Transformation<Identity> {
public:
  Identity(bool is_gpu = false) : isgpu(is_gpu) {
    transformation_(this, "identity")
      .setEntryLocation(is_gpu ? Device : Host)
      .input("source")
      .output("target")
      .types(Atypes::pass<0,0>)
      .func(&Identity::identity)
      ;
  };

  bool isgpu = false;

  void identity (Args args, Rets rets) {
    if (isgpu) { gpu_test(args, rets); }
    else rets[0].x = args[0].x;
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
