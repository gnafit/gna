#include "Sqrt.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

#include <Eigen/Core>

/**
 * @brief Constructor.
 */
Sqrt::Sqrt() {
    this->transformation_("sqrt")
        .input("points")
        .output("result")
        .types(new PassTypeT<double>(0, {0,-1}))
        .func(&Sqrt::calculate)
      ;
}

Sqrt::Sqrt(OutputDescriptor& output) : Sqrt() {
    transformations.front().inputs.front().connect(output);
}

/**
 * @brief Calculate the value of function.
 */
void Sqrt::calculate(FunctionArgs& fargs){
    fargs.rets[0].x = fargs.args[0].x.sqrt();
}
