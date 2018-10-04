#pragma once

#include "GNAObject.hh"

/**
 * @brief Calculate the element-wise ratio of two inputs.
 *
 * Inputs:
 *   - `ratio.top` -- the nominator.
 *   - `ratio.bottom` -- the denominator.
 *
 * Outputs:
 *   - `ratio.ratio` -- the result of a ratio.
 *
 * @author Maxim Gonchar
 * @date 2018
 */
class Ratio: public GNASingleObject,
             public TransformationBind<Ratio> {
public:
  Ratio();                                     ///< Constructor.
};
