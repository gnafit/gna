#pragma once

#include "GNAObject.hh"


namespace GNA {
  namespace GNAObjectTemplates {
    /**
     * @brief Calculate the element-wise product of the inputs.
     *
     * Outputs:
     *   - `product.product` -- the result of a product.
     *
     * @author Dmitry Taychenachev
     * @date 2015
     */
    template<typename FloatType>
    class ProductT: public GNASingleObjectT<FloatType,FloatType>,
                    public TransformationBind<ProductT<FloatType>, FloatType, FloatType> {

    private:
      using BaseClass = GNAObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
      using typename BaseClass::SingleOutput;
      using typename BaseClass::OutputDescriptor;

      ProductT();                                                   ///< Constructor
      ProductT(const typename OutputDescriptor::OutputDescriptors& outputs); ///< Construct Product from vector of outputs

      /**
       * @brief Construct a Product of two outputs
       * @param data1 -- first SingleOutput instance.
       * @param data2 -- second SingleOutput instance.
       */
      ProductT(SingleOutput& data1, SingleOutput& data2) : ProductT<FloatType>() {
        multiply(data1, data2);
      }

      /**
       * @brief Construct a Product of three outputs
       * @param data1 -- first SingleOutput instance.
       * @param data2 -- second SingleOutput instance.
       * @param data3 -- third SingleOutput instance.
       */
      ProductT(SingleOutput& data1, SingleOutput& data2, SingleOutput& data3) : ProductT<FloatType>() {
        multiply(data1, data2, data3);
      }

      InputDescriptorT<FloatType,FloatType> add_input(const char* name);  ///< Add an input by name and leave unconnected.

      /** @brief Add an input by name and leave unconnected. */
      InputDescriptorT<FloatType,FloatType> multiply(const char* name){
        return add_input(name);
      }

      InputDescriptorT<FloatType,FloatType> multiply(SingleOutput &data); ///< Add an input and connect it to the output.

      /**
       * @brief Add two inputs and connect them to the outputs.
       *
       * @param data1 -- first SingleOutput instance.
       * @param data2 -- second SingleOutput instance.
       * @return InputDescriptor instance for the last created input.
       */
      InputDescriptorT<FloatType, FloatType> multiply(SingleOutput &data1, SingleOutput &data2){
        multiply(data1);
        return multiply(data2);
      }

      /**
       * @brief Add three inputs and connect them to the outputs.
       *
       * @param data1 -- first SingleOutput instance.
       * @param data2 -- second SingleOutput instance.
       * @param data3 -- third SingleOutput instance.
       * @return InputDescriptor instance for the last created input.
       */
      InputDescriptorT<FloatType,FloatType> multiply(SingleOutput &data1, SingleOutput &data2, SingleOutput &data3){
        multiply(data1);
        return multiply(data2, data3);
      }
#ifdef GNA_CUDA_SUPPORT
      void product_ongpu(FunctionArgs& fargs);
#endif
    };
  }
}
