#pragma once

#include "GNAObject.hh"
#include "TypesFunctions.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    /**
     * @brief Calculate the element-wise sum of the inputs.
     *
     * Outputs:
     *   - `sum.sum` -- the result of a sum.
     *
     * @author Dmitry Taychenachev
     * @date 2015
     */
    template<typename FloatType>
    class SumT: public GNASingleObjectT<FloatType,FloatType>,
               public TransformationBind<SumT<FloatType>, FloatType, FloatType> {
    private:
      using BaseClass = GNASingleObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;

      SumT();                                                   ///< Constructor.
      SumT(const OutputDescriptor::OutputDescriptors& outputs); ///< Construct Sum from vector of outputs

      InputDescriptorT<FloatType, FloatType> add_input(const char* name);  ///< Add an input by name and leave unconnected.

      /** @brief Add an input by name and leave unconnected. */
      InputDescriptorT<FloatType,FloatType> add(const char* name){
        return add_input(name);
      }

      InputDescriptorT<FloatType,FloatType> add(SingleOutput &data);   ///< Add an input and connect it to the output.
      void sum_ongpu(FunctionArgs& fargs);
    };
  }
}

using Sum = GNA::GNAObjectTemplates::SumT<double>;
