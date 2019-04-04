#pragma once

#include "GNAObject.hh"
#include <vector>
#include <string>

namespace GNA{
      namespace GNAObjectTemplates{
            template<typename FloatType>
            class ArraySumT: public GNASingleObjectT<FloatType,FloatType>,
                             public TransformationBind<ArraySumT<FloatType>,FloatType,FloatType> {
            private:
                  using GNAObjectType = GNAObjectT<FloatType,FloatType>;
                  using typename GNAObjectType::FunctionArgs;
                  using typename GNAObjectType::TypesFunctionArgs;

            public:
                  using ArraySumType = ArraySumT<FloatType>;
                  using typename GNAObjectType::SingleOutput;

                  ArraySumT();
                  ArraySumT(SingleOutput& out);
                  ArraySumT(const std::string& name, SingleOutput& out);

            private:
                  void initialize(const std::string& name);
                  void sum(FunctionArgs& fargs);
                  void check(TypesFunctionArgs& fargs);

                  std::string          m_output_name;
                  dependant<FloatType> m_accumulated;
            };
      }
}
