#include "InterpLinear.hh"

double InterpLinear::interpolation_formula(double x, double y, double k, double point) const noexcept {
    return k * (point - x) + y;
}

Eigen::ArrayXd InterpLinear::compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept {
    return (ys.tail(nseg)-ys.head(nseg))/widths;          /// k coefficient
}
