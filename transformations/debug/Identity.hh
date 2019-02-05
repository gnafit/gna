#pragma once

#include <iostream>
#include "GNAObject.hh"
#include "TypesFunctions.hh"

//
// Identity transformation
//

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class IdentityT: public GNASingleObjectT<FloatType,FloatType>,
               	     public TransformationBind<IdentityT<FloatType>, FloatType, FloatType> {
    private:
      using BaseClass = GNASingleObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;

      IdentityT();
      void dump();
    };
  }
}
