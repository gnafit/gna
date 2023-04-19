#include "Exp.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

#include <Eigen/Core>

/**
 * @brief Constructor.
 */
Exp::Exp() {
    this->transformation_("exp")
        .input("points")
        .output("result")
        .types(new CheckKindT<double>(DataKind::Points), new PassTypeT<double>(0, {0,-1}))
        .func(&Exp::calculate)
      ;
}

Exp::Exp(OutputDescriptor& output) : Exp() {
    transformations.front().inputs.front().connect(output);
}

/**
 * @brief Calculate the value of function.
 */
void Exp::calculate(FunctionArgs& fargs){
    fargs.rets[0].x = fargs.args[0].x.exp();
}
