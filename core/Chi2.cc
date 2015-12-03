#include "Chi2.hh"

using namespace Eigen;

void Chi2::calculateChi2(Args args, Rets rets) {
  VectorXd diff = args[0].vec - args[1].vec;
  args[2].mat.triangularView<Eigen::Lower>().solveInPlace(diff);
  rets[0].arr(0) = diff.array().square().sum();
}
