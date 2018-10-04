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
  Ratio();                                        ///< Default constructor.
  Ratio(SingleOutput& top, SingleOutput& bottom); ///< Construct ratio of top and bottom

  OutputDescriptor divide(SingleOutput& top, SingleOutput& bottom); ///< Bind nomenator, denomenator and return the ratio (output)
};
