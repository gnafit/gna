#pragma once

#include "GNAObject.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    /**
     * @brief Calculate the element-wise sum of squares of the inputs.
     *
     * Outputs:
     *   - `sumsq.sumsq` -- the result of a sumsq.
     *
     * @author Maxim Gonchar
     * @date 2020
     */
    template<typename FloatType>
    class SumSqT: public GNASingleObjectT<FloatType,FloatType>,
               public TransformationBind<SumSqT<FloatType>, FloatType, FloatType> {
    private:
      using BaseClass = GNAObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
      using typename BaseClass::SingleOutput;
      using typename BaseClass::OutputDescriptor;

      SumSqT();                                                            ///< Constructor.
      SumSqT(const typename OutputDescriptor::OutputDescriptors& outputs); ///< Construct SumSq from vector of outputs

      InputDescriptorT<FloatType, FloatType> add_input(const char* name);  ///< Add an input by name and leave unconnected.

      /** @brief Add an input by name and leave unconnected. */
      InputDescriptorT<FloatType,FloatType> add(const char* name){
        return add_input(name);
      }

      InputDescriptorT<FloatType,FloatType> add(SingleOutput &data);   ///< Add an input and connect it to the output.

#ifdef GNA_CUDA_SUPPORT
      void sumsq_ongpu(FunctionArgs& fargs);
#endif
    };
  }
}

