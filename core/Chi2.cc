#include <boost/lexical_cast.hpp>

#include "Chi2.hh"
#include "test.h"
#include "simp.h"

using namespace Eigen;

void Chi2::add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov) {
  t_["chi2"].input(theory);
  t_["chi2"].input(data);
  t_["chi2"].input(cov);
}

void Chi2::checkTypes(Atypes args, Rtypes rets) {
  if (args.size()%3 != 0) {
    throw args.undefined();
  }
  for (size_t i = 0; i < args.size(); i+=3) {
    if (args[i+0].shape.size() != 1) {
      throw rets.error(rets[0], "non-vector theory");
    }

    if (args[i+1].shape != args[i+0].shape) {
      throw rets.error(rets[0], "data and theory have different shape");
    }
    if (args[i+2].shape.size() != 2 ||
        args[i+2].shape[0] != args[i+2].shape[1]) {
      throw rets.error(rets[0], "incompatible covmat shape");
    }
    if (args[i+2].shape[0] != args[i+0].shape[0]) {
      throw rets.error(rets[0], "incompatible covmat shape 2");
    }
  }
  rets[0] = DataType().points().shape(1);
}

void Chi2::calculateChi2(Args args, Rets rets) {
  rets[0].arr(0) = 0;
  std::cout << "I AM HERE!" << std::endl;
  simpfun();
  for (size_t i = 0; i < args.size(); i+=3) {
    VectorXd diff = args[i+0].vec - args[i+1].vec;
    args[i+2].mat.triangularView<Eigen::Lower>().solveInPlace(diff);
    rets[0].arr(0) += diff.array().square().sum();
  }
}
