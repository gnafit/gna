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
  Identity() {
    transformation_("identity")
      .input("source")
      .output("target")
      .types(TypesFunctions::pass<0,0>)
      .func([](FunctionArgs& fargs){ fargs.rets[0].x = fargs.args[0].x;})
#ifdef GNA_CUDA_SUPPORT     //
      .setEntryLocation(DataLocation::Device)
      .func("GPU", &Identity::gpu_test)
#endif
      ;
  };

#ifdef GNA_CUDA_SUPPORT
  void gpu_test (FunctionArgs& fargs) {
    //this.setEntryLocation(DataLocation::Device);
    fargs.rets[0].gpuArr->setByDeviceArray(fargs.args[0].gpuArr->devicePtr);
    *(fargs.rets[0].gpuArr) *=15;
    std::cout << "Dump: ";
    fargs.rets[0].gpuArr->dump();
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
