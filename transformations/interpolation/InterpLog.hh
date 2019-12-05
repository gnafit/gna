#pragma once

#include "InterpBase.hh"


class InterpLog: public InterpBase {
      InterpLog() : InterpBase() {};                                                                                                ///< Constructor.
      InterpLog(SingleOutput& x, SingleOutput& newx)                  : InterpBase(x, newx) {};                                                             ///< Constructor.
      InterpLog(SingleOutput& x, SingleOutput& y, SingleOutput& newx) : InterpBase(x, y, newx) {};                                            ///< Constructor.

      virtual double interpolation_formula(double x, double y, double k, double point) const noexcept final;
      virtual Eigen::ArrayXd compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept final;
};
