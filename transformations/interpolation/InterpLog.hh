#pragma once

#include "InterpBase.hh"


class InterpLog: public InterpBase {
public:
      InterpLog() : InterpBase() {};                                                                                                ///< Constructor.
      InterpLog(SingleOutput& x, SingleOutput& newx)                  : InterpBase(x, newx) {};                                                             ///< Constructor.
      InterpLog(SingleOutput& x, SingleOutput& y, SingleOutput& newx) : InterpBase(x, y, newx) {};                                            ///< Constructor.

private:
      double interpolation_formula(double x, double y, double k, double point) const noexcept final;
      double interpolation_formula_below(double x, double y, double k, double point) const noexcept final;
      double interpolation_formula_above(double x, double y, double k, double point) const noexcept final;
      Eigen::ArrayXd compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept final;
};
