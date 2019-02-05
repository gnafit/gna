#pragma once

#include "GNAObjectBindMN.hh"

/**
 * @brief Calculate the element-wise sum of the inputs.
 *
 * Outputs:
 *   - `sum.sum` -- the result of a sum.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class MultiSum: public GNAObjectBindMN,
                public TransformationBind<MultiSum> {
public:
    MultiSum();                                                   ///< Constructor.
    MultiSum(const OutputDescriptor::OutputDescriptors& outputs); ///< Construct MultiSum from vector of outputs

    TransformationDescriptor add_transformation(const std::string& name=""); ///< Add new transformation
};
