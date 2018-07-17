#include "Exp.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>

/**
 * @brief Constructor.
 */
Exp::Exp() {
    transformation_("exp")
        .input("points")
	.output("result")
	.types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
	.func(&Exp::calculate)
      ;
}

/**
 * @brief Calculate the value of function.
 */
void Exp::calculate(Args args, Rets rets){
    rets[0].x = args[0].x.exp();
}

