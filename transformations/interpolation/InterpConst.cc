#include "InterpConst.hh"

double InterpConst::interpolation_formula(double x, double y, double k, double point) const noexcept {
    return y;
}

double InterpConst::interpolation_formula_below(double x, double y, double k, double point) const noexcept {
    return y;
}

double InterpConst::interpolation_formula_above(double x, double y, double k, double point) const noexcept {
    return k;
}

Eigen::ArrayXd InterpConst::compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept {
    auto size=xs.size()-1;
    auto ret=Eigen::ArrayXd(size);
    ret[size-1]=ys[size];
    return ret;
}
