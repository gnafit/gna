#pragma once

#include "GNAObject.hh"

/**
 * @brief Calculate the element-wise ratio of two inputs.
 *
 * Broadcastable.
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
class RatioBC: public GNASingleObject,
               public TransformationBind<RatioBC> {
public:
  RatioBC();                                        ///< Default constructor.
  RatioBC(SingleOutput& top, SingleOutput& bottom); ///< Construct ratio of top and bottom

  OutputDescriptor divide(SingleOutput& top, SingleOutput& bottom); ///< Bind nomenator, denomenator and return the ratio (output)
};
