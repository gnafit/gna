#include "SelfPower.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>
#include <cmath>

/**
 * @brief Constructor.
 *
 * Creates two transformations for functions with positive and negative power.
 *
 * @param scalename="sp_scale" -- name of a variable to scale the argument.
 */
SelfPower::SelfPower(const char* scalename/*="sp_scale"*/) {
    variable_(&m_scale, scalename);

    transformation_("selfpower")
        .input("points")
	.output("result")
	.types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
	.func(&SelfPower::calculate)
      ;

    transformation_("selfpower_inv")
        .input("points")
	.output("result")
	.types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
	.func(&SelfPower::calculate_inv)
      ;
}

/**
 * @brief Calculate the value of function with positive power.
 */
void SelfPower::calculate(Args args, Rets rets){
    auto& res = rets[0].x = args[0].x/m_scale.value();
    res=res.pow(res);
}

/**
 * @brief Calculate the value of function with negative power.
 */
void SelfPower::calculate_inv(Args args, Rets rets){
    auto& res = rets[0].x = args[0].x/m_scale.value();
    res=res.pow(-res);
}
