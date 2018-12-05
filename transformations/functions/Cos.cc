#include "Cos.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>

/**
 * @brief Constructor.
 */
Cos::Cos() {
    transformation_("cos")
        .input("points")
	.output("result")
	.types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
	.func(&Cos::calculate)
      ;
}

Cos::Cos(OutputDescriptor& output) : Cos() {
    transformations.front().inputs.front().connect(output);
}

/**
 * @brief Calculate the value of function.
 */
void Cos::calculate(FunctionArgs& fargs){
    fargs.rets[0].x = fargs.args[0].x.cos();
}

