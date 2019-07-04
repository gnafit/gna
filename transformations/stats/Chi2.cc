#include <boost/lexical_cast.hpp>

#include "Chi2.hh"

using namespace Eigen;

void Chi2::add(SingleOutput &theory, SingleOutput &data, SingleOutput &errors) {
  t_["chi2"].input(theory);
  t_["chi2"].input(data);
  t_["chi2"].input(errors);
}

void Chi2::checkTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (args.size()%3 != 0) {
    throw args.undefined();
  }
  for (size_t i = 0; i < args.size(); i+=3) {
    auto& theory = args[i+0];
    auto& data   = args[i+1];
    auto& errors = args[i+2];
    if (theory.shape.size() != 1) {
      throw rets.error(rets[0], "non-vector theory");
    }
    if (data.shape != theory.shape) {
      throw rets.error(rets[0], "data and theory have different shape");
    }

    switch(errors.shape.size()){
      case 1:
        /// Errors: uncorrelated uncertainties (first power)
        break;
      case 2:
        /// Errors: L - lower triangular decomposition of covariance matrix
        if (errors.shape[0] != errors.shape[1]) {
          throw rets.error(rets[0], "incompatible covmat shape");
        }
        break;
      default:
        throw rets.error(rets[0], "invalid dimension (errors input)");
        break;
    }

    if (errors.shape[0] != theory.shape[0]) {
      throw rets.error(rets[0], "errors are unconsistent with data");
    }
  }
  rets[0] = DataType().points().shape(1);
}

void Chi2::calculateChi2(FunctionArgs fargs) {
  auto& args=fargs.args;
  double res=0.0;
  for (size_t i = 0; i < args.size(); i+=3) {
    VectorXd diff = args[i+0].vec - args[i+1].vec;
    auto& errors = args[i+2];
    switch(errors.type.shape.size()){
      case 1:
        diff.array()/=errors.arr;
        break;
      case 2:
        errors.mat.triangularView<Eigen::Lower>().solveInPlace(diff);
        break;
      default:
        break;
    }
    res += diff.array().square().sum();
  }
  fargs.rets[0].arr(0)=res;
}
