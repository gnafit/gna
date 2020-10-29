#include "InterpConst.hh"

double InterpConst::interpolation_formula(double x, double y, double k, double point) const noexcept {
    return y;
}

Eigen::ArrayXd InterpConst::compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept {
    return Eigen::ArrayXd(xs.size()-1);
}
