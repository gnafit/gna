#pragma once

#include "GNAObject.hh"

/**
 * @brief Calculate the element-wise product of the inputs.
 *
 * Outputs:
 *   - `product.product` -- the result of a product.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class Product: public GNASingleObject,
               public TransformationBind<Product> {
public:
  Product();                                                   ///< Constructor
  Product(const OutputDescriptor::OutputDescriptors& outputs); ///< Construct Product from vector of outputs

  /**
   * @brief Construct a Product of two outputs
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   */
  Product(SingleOutput& data1, SingleOutput& data2) : Product() {
    multiply(data1, data2);
  }

  /**
   * @brief Construct a Product of three outputs
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   * @param data3 -- third SingleOutput instance.
   */
  Product(SingleOutput& data1, SingleOutput& data2, SingleOutput& data3) : Product() {
    multiply(data1, data2, data3);
  }

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
  InputDescriptor multiply(SingleOutput &data1, SingleOutput &data2){ multiply(data1); return multiply(data2); }

  /**
   * @brief Add three inputs and connect them to the outputs.
   *
   * @param data1 -- first SingleOutput instance.
   * @param data2 -- second SingleOutput instance.
   * @param data3 -- third SingleOutput instance.
   * @return InputDescriptor instance for the last created input.
   */
  InputDescriptor multiply(SingleOutput &data1, SingleOutput &data2, SingleOutput &data3){ multiply(data1); return multiply(data2, data3); }

};
