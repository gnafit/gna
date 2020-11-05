#pragma once

#include "GNAObject.hh"

/**
 * @brief Calculate the element-wise product of the inputs. Depending on condition may multiply different number of inputs.
 *
 * In case condition=0 calculates a product only of m_nprod elements. Calculates the full product otherwise.
 *
 * Outputs:
 *   - `product.product` -- the result of a product.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class ConditionalProduct: public GNASingleObject,
                          public TransformationBind<ConditionalProduct> {

public:
  ConditionalProduct(size_t nprod, std::string condition); ///< Constructor
  ConditionalProduct(size_t nprod, std::string condition, const typename OutputDescriptor::OutputDescriptors& outputs); ///< Construct Product from vector of outputs

  /**
   * @brief Construct a Product of two outputs
   * @param condition -- name of the condition variable.
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   */
  ConditionalProduct(std::string condition, SingleOutput& data1, SingleOutput& data2) : ConditionalProduct(1u, condition) {
    multiply(data1, data2);
  }

  void compute_product(FunctionArgs& fargs);    ///< Calculation function

  InputDescriptor add_input(const char* name);  ///< Add an input by name and leave unconnected.

  /** @brief Add an input by name and leave unconnected. */
  InputDescriptor multiply(const char* name){
    return add_input(name);
  }

  InputDescriptor multiply(SingleOutput &data); ///< Add an input and connect it to the output.

  /**
   * @brief Add two inputs and connect them to the outputs.
   *
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   * @return InputDescriptor instance for the last created input.
   */
  InputDescriptor multiply(SingleOutput &data1, SingleOutput &data2){
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
  InputDescriptor multiply(SingleOutput &data1, SingleOutput &data2, SingleOutput &data3){
    multiply(data1);
    return multiply(data2, data3);
  }

private:
  size_t m_nprod;               ///< Number of elements to multiply in case m_condition!=0.
  variable<double> m_condition; ///< Condition to test.
};
