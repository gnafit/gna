#include <boost/lexical_cast.hpp>
#include "Poisson.hh"
#include "TMath.h"
#include <Eigen/Core>
#include <cmath>

using namespace Eigen;


void Poisson::add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov) {
  t_["poisson"].input(theory);
  t_["poisson"].input(data);
  t_["poisson"].input(cov);
}


void Poisson::checkTypes(Atypes args, Rtypes rets) {
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
	else 		return 0;
}

void Poisson::calcPoissonApprox(Args args, Rets rets) {

/***************************************************************************
*       Formula: log of Poisson
*
*        -2 * ln(Poisson) =
*	        -2 * sum(data_i * log(theory_j) -  theory_j  - ln data_i! ) 
*
****************************************************************************/

	rets[0].arr(0) = 0;
	for (size_t i = 0; i < args.size(); i+=3) {
            rets[0].arr(0) += (args[i+0].arr.log() * args[i+1].arr
		   	      - args[i+0].arr).sum();
 	    rets[0].arr(0) -= (args[i+1].arr).unaryExpr(&lnFactorialApprox).sum();
	}
	rets[0].arr(0) *= -2;
}

void Poisson::calcPoisson(Args args, Rets rets) {

/***************************************************************************
*       Formula: log of Poisson
*
*        -2 * ln(Poisson) =
*               -2 * sum(data_i * log(theory_j) -  theory_j  - ln data_i! ) 
*
****************************************************************************/
        rets[0].arr(0) = 0;
	for (size_t i = 0; i < args.size(); i+=3) {
            rets[0].arr(0) += (args[i+0].arr.log() * args[i+1].arr
                            - args[i+0].arr
                            - (args[i+1].arr).unaryExpr(&addOneInLnGamma) )
                            .sum();
        }
        rets[0].arr(0) *= -2;
}

