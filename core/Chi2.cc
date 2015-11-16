#include "Chi2.hh"

using namespace Eigen;

void Chi2::calculateChi2(Args args, Rets rets) {
  MatrixXd diff = (args[0].x - args[1].x).matrix();
  rets[0].x = diff.transpose() * args[2].as2d().matrix() * diff;
}
