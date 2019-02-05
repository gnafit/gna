#pragma once

#include "GNAObject.hh"

/**
 * @brief Calculate the element-wise sum of the inputs.
 *
 * Outputs:
 *   - `sum.sum` -- the result of a sum.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class MultiSum: public GNAObject,
                public TransformationBind<MultiSum> {
public:
  MultiSum();                                                   ///< Constructor.
  MultiSum(const OutputDescriptor::OutputDescriptors& outputs); ///< Construct MultiSum from vector of outputs

  InputDescriptor add_input(const char* name);  ///< Add an input by name and leave unconnected.

  /** @brief Add an input by name and leave unconnected. */
  InputDescriptor add(const char* name){
    return add_input(name);
  }

  InputDescriptor add(SingleOutput &data);   ///< Add an input and connect it to the output.
};
