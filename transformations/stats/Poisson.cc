#include <boost/lexical_cast.hpp>
#include "Poisson.hh"
#include "TMath.h"
#include <Eigen/Core>
#include <cmath>

using namespace Eigen;

Poisson::Poisson(bool ln_approx) {
  transformation_("poisson")
    .output("poisson")
    .types(&Poisson::checkTypes)
    .func(ln_approx ? &Poisson::calcPoissonApprox : &Poisson::calcPoisson)
    ;
  m_transform = t_["poisson"];
}

void Poisson::add(SingleOutput &theory, SingleOutput &data) {
  t_["poisson"].input(theory);
  t_["poisson"].input(data);
}


void Poisson::checkTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (args.size()%2 != 0) {
    throw args.undefined();
  }
  for (size_t i = 0; i < args.size(); i+=2) {
    if (args[i+0].shape.size() != 1) {
      throw rets.error(rets[0], "non-vector theory");
    }

    if (args[i+1].shape != args[i+0].shape) {
      throw rets.error(rets[0], "data and theory have different shape");
    }
  }
  rets[0] = DataType().points().shape(1);
}

double addOneInLnGamma(double in)
{
  return TMath::LnGamma(in + 1);
}

double lnFactorialApprox(double x)
{
  if (!(x == 0.0 || x == 1.0))
  {
    return x * std::log(x);
  }
  else
    return 0;
}

void Poisson::calcPoissonApprox(FunctionArgs fargs) {
  /***************************************************************************
   *       Formula: log of Poisson
   *
   *        -2 * ln(Poisson) =
   *            -2 * sum(data_i * log(theory_j) -  theory_j  - ln data_i! )
   *
   ****************************************************************************/
  auto& args=fargs.args;

  double res(0.0);
  for (size_t i = 0; i < args.size(); i+=2) {
    auto& theory=args[i+0].arr;
    auto& data=args[i+1].arr;
    res += (data*theory.log() - theory - data.unaryExpr(&lnFactorialApprox)).sum();
  }
  fargs.rets[0].arr(0) = -2*res;
}

void Poisson::calcPoisson(FunctionArgs fargs) {
  /***************************************************************************
   *       Formula: log of Poisson
   *
   *        -2 * ln(Poisson) =
   *               -2 * sum(data_i * log(theory_j) -  theory_j  - ln data_i! )
   *
   ****************************************************************************/
  auto& args=fargs.args;

  double res(0.0);
  for (size_t i = 0; i < args.size(); i+=2) {
    auto& theory=args[i+0].arr;
    auto& data=args[i+1].arr;
    res += (data*theory.log() - theory - data.unaryExpr(&addOneInLnGamma) ).sum();
  }
  fargs.rets[0].arr(0) = -2*res;
}

