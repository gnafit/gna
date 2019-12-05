#pragma once

#include "InterpBase.hh"

/**
 * @brief Exponential interpolation (unordered).
 *
 * For a given `x`, `y` and `newx` computes `newy` via exponential:
 *   - f(x) = k_i * exp( -(x-x_i)*b_i ) for x in [x_i, x_i+1)
 *   - k = f(x_0)
 *   - b is chosen such that function was continuous:
 *   - b_i = - ln(k_i+1 / k_i) / ( x_i+1 - x_i )
 *
 * The object inherits InSegment that determines the segments to be used for interpolation.
 *
 * @note No check is performed on the value of `b`.
 *
 * Inputs:
 *   - interp.newx -- array with points with fine steps to interpolate on.
 *   - interp.x -- array with coarse points.
 *   - interp.y -- values of a function on `x`.
 *   - interp.insegment -- segments of `newx` in `x`. See InSegment.
 *
 * Outputs:
 *   - interp.interp
 *
 * The connection may be done via InterpExpo::interpolate() method.
 *
 * @author Maxim Gonchar, updated by Konstantin Treskov
 * @date 02.2017
 */

class InterpExpo: public InterpBase {
public:
      InterpExpo() : InterpBase() {};                                                                                                ///< Constructor.
      InterpExpo(SingleOutput& x, SingleOutput& newx)                  : InterpBase(x, newx) {};                                                             ///< Constructor.
      InterpExpo(SingleOutput& x, SingleOutput& y, SingleOutput& newx) : InterpBase(x, y, newx) {};                                            ///< Constructor.

private:
      virtual double interpolation_formula(double x, double y, double k, double point) const noexcept final;
      virtual Eigen::ArrayXd compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept final;
};
