#pragma once

#include <iostream>
#include "GNAObject.hh"

//
// Identity transformation
//

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class IdentityT: public GNASingleObjectT<FloatType,FloatType>,
               	     public TransformationBind<IdentityT<FloatType>, FloatType, FloatType> {
    private:
      using BaseClass = GNAObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;

      IdentityT();
      void dump();

      void identity_gpu_h(FunctionArgs& fargs);
      void identity_gpu_d(FunctionArgs& fargs);
    };
  }
}
