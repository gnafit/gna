#include "Sin.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>

/**
 * @brief Constructor.
 */
Sin::Sin() {
    transformation_("sin")
        .input("points")
	.output("result")
	.types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
	.func(&Sin::calculate)
      ;
}

/**
 * @brief Calculate the value of function.
 */
void Sin::calculate(FunctionArgs& fargs){
    fargs.rets[0].x = fargs.args[0].x.sin();
}

