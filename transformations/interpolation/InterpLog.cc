#include "InterpLog.hh"
#include <cmath>

double InterpLog::interpolation_formula(double x, double y, double k, double point) const noexcept {
      auto b = std::exp(y);
      return std::log(k*(point - x) + b);
}

double InterpLog::interpolation_formula_above(double x, double y, double k, double point) const noexcept {
      auto b = std::exp(y);
      return std::log(k*(point - x) + b);
}

double InterpLog::interpolation_formula_below(double x, double y, double k, double point) const noexcept {
      auto b = std::exp(y);
      return std::log(k*(point - x) + b);
}

Eigen::ArrayXd InterpLog::compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept {
    auto  b_a = ys.exp();
    return (b_a.tail(nseg)-b_a.head(nseg))/widths;          /// k coefficient
}
