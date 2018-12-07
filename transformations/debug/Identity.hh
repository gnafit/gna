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
  Identity(){
    transformation_("identity")
      .input("source")
      .output("target")
      .types(TypesFunctions::pass<0,0>)
      .func([](FunctionArgs& fargs){ fargs.rets[0].x = fargs.args[0].x; })
      ;
  };

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
