#include "InterpLogx.hh"
#include <cmath>

double InterpLogx::interpolation_formula(double x, double y, double k, double point) const noexcept {
       return k * std::log(point/x) + y;
}

Eigen::ArrayXd InterpLogx::compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept {
    auto logratios=(xs.tail(nseg)/xs.head(nseg)).log();
    return (ys.tail(nseg)-ys.head(nseg))/logratios;
}
