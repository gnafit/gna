#include "InterpExpo.hh"
#include <cmath>

double InterpExpo::interpolation_formula(double x, double y, double k, double point) const noexcept {
      return y * std::exp((x - point)*k);
}

Eigen::ArrayXd InterpExpo::compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept {
    return  (ys.head(nseg)/ys.tail(nseg)).log()/widths;
}
