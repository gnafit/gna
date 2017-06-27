#include <boost/lexical_cast.hpp>
#include "Poisson.hh"
#include "TMath.h"
#include <Eigen/Core>
#include <cmath>

using namespace Eigen;


void Poisson::add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov, bool ln_approx) {
  t_["poisson"].input(theory);
  t_["poisson"].input(data);
  t_["poisson"].input(cov);
  if (ln_approx)
  {
	approx = true;
  }
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

double NonZeroAdd(double x)
{
	if (!(x == 0.0 || x == 1.0)) 
	{	
		return x * std::log(x);
	}
	else 		return 0;
}

double NonZeroLog(double x)
{
	if (!(x == 0.0 || x == 1.0))
	{
		return std::log(x);
	}
	else
	{
                return 0;
        }

}


void Poisson::calcPoisson(Args args, Rets rets) {

/***************************************************************************
*       Formula: log of Poisson
*
*        -2 * ln(Poisson) =
*	        -2 * sum(data_i * log(theory_j) -  theory_j  - ln data_i! ) 
*
****************************************************************************/

	rets[0].arr(0) = 0;
	if (!approx)
	{
		for (size_t i = 0; i < args.size(); i+=3) {
        	    rets[0].arr(0) += (args[i+0].vec.array().unaryExpr(&NonZeroLog) * args[i+1].vec.array()
					- args[i+0].vec.array() 
					- (args[i+1].vec.array()).unaryExpr(&addOneInLnGamma) )
					.sum();
		}
	}
	else
	{

		for (size_t i = 0; i < args.size(); i+=3) {
            	    rets[0].arr(0) += (args[i+0].vec.array().unaryExpr(&NonZeroLog) * args[i+1].vec.array()
					- args[i+0].vec.array()).sum();
 		    rets[0].arr(0) -= (args[i+1].vec.array()).unaryExpr(&NonZeroAdd).sum();



		}
	}

	rets[0].arr(0) *= -2;
}
