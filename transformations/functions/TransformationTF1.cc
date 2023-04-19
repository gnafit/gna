#include "TransformationTF1.hh"
#include "TypesFunctions.hh"
#include <Eigen/Core>

/**
 * @brief Constructor.
 */
TransformationTF1::TransformationTF1(TF1* fcn): m_fcn(fcn) {
    assert(m_fcn);

    transformation_("tf1")
        .input("arg")
        .output("result")
        .types(TypesFunctions::ifPoints<0>, TypesFunctions::pass<0>)
        .func(&TransformationTF1::calculate)
      ;
}

TransformationTF1::TransformationTF1(TF1* fcn, OutputDescriptor& output) : TransformationTF1(fcn) {
    output >> transformations.front().inputs.front();
}

/**
 * @brief Calculate the value of function.
 */
void TransformationTF1::calculate(FunctionArgs& fargs){
    fargs.rets[0].x = fargs.args[0].x.unaryExpr([this](double v){return this->m_fcn->Eval(v);});
}

