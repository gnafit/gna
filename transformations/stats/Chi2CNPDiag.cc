#include "Chi2CNPDiag.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;
#include "TMath.h"
#include <Eigen/Core>
#include <cmath>

using namespace Eigen;

Chi2CNPDiag::Chi2CNPDiag() {
  transformation_("poisson")
    .output("poisson")
    .types(&Chi2CNPDiag::checkTypes)
    .func(&Chi2CNPDiag::calcPoisson)
    ;

  auto poisson = transformations["poisson"];
  auto out = transformations["poisson_const"].outputs[0];
  poisson.input(out);

  m_transform = t_["poisson"];
}

void Chi2CNPDiag::add(SingleOutput &theory, SingleOutput &data) {
  auto poisson_const = transformations["poisson_const"];
  poisson_const.input(data);

  auto poisson = transformations["poisson"];
  poisson.input(theory);
  poisson.input(data);
}

void Chi2CNPDiag::checkTypesConst(TypesFunctionArgs fargs) {
  auto& rets=fargs.rets;
  rets[0] = DataType().points().shape(1);
}

void Chi2CNPDiag::checkTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (args.size()%2 != 1) {
    throw args.undefined();
  }
  if (args[0].kind!=DataKind::Undefined && args[0].size() != 1) {
    throw rets.error(rets[0], "lnPoisson const term size should be 1");
  }
  for (size_t i = 1; i < args.size(); i+=2) {
    if (args[i+1].shape != args[i+0].shape) {
      throw rets.error(rets[0], "data and theory have different shape");
    }
  }
  rets[0] = DataType().points().shape(1);
}

double lnPi(double num)
{
  return TMath::LnGamma(num + 1);
}

double lnFactorialStirling(double x)
{
  if (!(x == 0.0 || x == 1.0))
  {
    return x * std::log(x) - x;
  }
  else
    return 0;
}

void Chi2CNPDiag::calcPoisson(FunctionArgs fargs) {
  /***************************************************************************
   *       Formula: log of Poisson
   *        -2 * ln(Poisson) =
   *             2 * sum(theory_j - data_i * log(theory_j) + ln data_i! )
   *
   ****************************************************************************/
  auto& args=fargs.args;

  double res=args[0].arr(0);
  for (size_t i = 1; i < args.size(); i+=2) {
    auto& theory=args[i+0].arr;
    auto& data=args[i+1].arr;
    res += (theory - data*theory.log()).sum();
  }
  fargs.rets[0].arr(0) = 2*res;
}

void Chi2CNPDiag::calcPoissonConstApprox(FunctionArgs fargs) {
  /***************************************************************************
   *       Formula: log of Poisson
   *        -2 * ln(Poisson) =
   *            -2 * sum(data_i * log(theory_j) -  theory_j  - ln data_i! )
   *
   *       Compute: ln data_i! ≈ data_i ln data_i - data_i
   *
   ****************************************************************************/
  auto& args=fargs.args;

  double res(0.0);
  for (size_t i = 0; i < args.size(); ++i) {
    res += (args[i].arr.unaryExpr(&lnFactorialStirling)).sum();
  }
  fargs.rets[0].arr(0) = res;
}

void Chi2CNPDiag::calcPoissonConst(FunctionArgs fargs) {
  /***************************************************************************
   *       Formula: log of Poisson
   *
   *        -2 * ln(Poisson) =
   *               -2 * sum(data_i * log(theory_j) -  theory_j  - ln data_i! )
   *
   *       Compute: ln data_i!
   *
   ****************************************************************************/
  auto& args=fargs.args;

  double res(0.0);
  for (size_t i = 0; i < args.size(); ++i) {
    res += args[i].arr.unaryExpr(&lnPi).sum();
  }
  fargs.rets[0].arr(0) = res;
}

