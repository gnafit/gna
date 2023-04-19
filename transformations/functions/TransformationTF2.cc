#include "TransformationTF2.hh"
#include "TypeClasses.hh"
#include <Eigen/Core>

using namespace TypeClasses;

/**
 * @brief Constructor.
 */
TransformationTF2::TransformationTF2(TF2* fcn): m_fcn(fcn) {
    assert(m_fcn);

    transformation_("tf2")
        .input("arg0")
        .input("arg1")
        .output("result")
        .types(new CheckSameTypesT<double>({0,1}, "shape"), TypesFunctions::pass<0>)
        .func(&TransformationTF2::calculate)
      ;
}

TransformationTF2::TransformationTF2(TF2* fcn, OutputDescriptor& xoutput, OutputDescriptor& youtput) :
TransformationTF2(fcn)
{
    auto& inputs=transformations.front().inputs;
    xoutput >> inputs[0];
    youtput >> inputs[1];
}

/**
 * @brief Calculate the value of function.
 */
void TransformationTF2::calculate(FunctionArgs& fargs){
    auto& args=fargs.args;
    auto* arg0b=args[0].x.data();
    auto* arg1b=args[1].x.data();
    auto& ret=fargs.rets[0].x;
    auto* retb=ret.data();
    for (int i = 0; i < ret.size(); ++i) {
        *retb = m_fcn->Eval(*arg0b, *arg1b);
        ++retb;
        ++arg0b;
        ++arg1b;
    }
}

