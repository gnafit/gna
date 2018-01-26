#ifndef IDENTITYX2_H
#define IDENTITYX2_H 1

#include <stdio.h>

//
// Identity transformation
//
class IdentityX2: public GNAObject,
                public Transformation<IdentityX2> {
public:
  IdentityX2(){
    transformation_(this, "identityx2")
      .input("source")
      .output("target")
      .types(Atypes::pass<0,0>)
      .func([](Args args, Rets rets){ 
		rets[0].x = args[0].x + args[0].x; 
	})
      ;
  };

  void dump(){
      auto& data = t_["identityx2"][0];

      if( data.type.shape.size()==2u ){
          std::cout<<data.arr2d<<std::endl;
      }
      else{
          std::cout<<data.arr<<std::endl;
      }
  }
};

#endif
