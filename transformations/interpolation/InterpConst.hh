#pragma once

#include "InterpBase.hh"

/**
 * @brief Const interpolation (unordered).
 *
 * For a given `x`, `y` and `newx` computes `newy` via linear approximation:
 *   - f(x) = y_i, x ∈ (x_i, x_{i+1})
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
 * The connection may be done via InterpConst::interpolate() method.
 *
 * @author Maxim Gonchar, updated by Konstantin Treskov
 * @date 02.2017
 */

class InterpConst: public InterpBase {
public:
      InterpConst() : InterpBase() {};                                                                                                ///< Constructor.
      InterpConst(SingleOutput& x, SingleOutput& newx)                  : InterpBase(x, newx) {};                                                             ///< Constructor.
      InterpConst(SingleOutput& x, SingleOutput& y, SingleOutput& newx) : InterpBase(x, y, newx) {};                                            ///< Constructor.

private:
      double interpolation_formula(double x, double y, double k, double point) const noexcept final;
      double interpolation_formula_below(double x, double y, double k, double point) const noexcept final;
      double interpolation_formula_above(double x, double y, double k, double point) const noexcept final;
      Eigen::ArrayXd compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept final;
};
