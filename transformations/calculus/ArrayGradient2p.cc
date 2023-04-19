#include "ArrayGradient2p.hh"
#include "TypeClasses.hh"

#include <stdexcept>

ArrayGradient2p::ArrayGradient2p()
{
    using namespace TypeClasses;

    transformation_("gradient")
        .input("x")
        .input("y")
        .output("xout")
        .output("gradient")
        .types(new CheckSameTypesT<double>({0,-1}, "shape"))
        .types(new CheckNdimT<double>(1))
        .types(&ArrayGradient2p::types)
        .func(&ArrayGradient2p::calc_gradient);
}

OutputDescriptor ArrayGradient2p::inputs(SingleOutput& x, SingleOutput& y){
    const auto& t = t_[0];
    const auto& inputs = t.inputs();
    inputs[0].connect(x.single());
    inputs[1].connect(y.single());
    return OutputDescriptor(t.outputs()[0]);
}

void ArrayGradient2p::types(TypesFunctionArgs& fargs){
    auto& ytype = fargs.args[1];
    if(!ytype.defined()){
        return;
    }

    size_t nin = ytype.shape[0];
    if(nin<2){
        throw fargs.args.error(ytype, "ArrayGradient2p: input is expected to have at least 2 points.");
    }
    fargs.rets[0].points().shape(nin-1u);
    fargs.rets[1].points().shape(nin-1u);
}

void ArrayGradient2p::calc_gradient(FunctionArgs& fargs){
    // implements numpy.gradient technique
    auto& args = fargs.args;
    auto& rets = fargs.rets;

    auto& X = args[0];
    auto& Y = args[1];

    auto& Xout = rets[0];
    auto& Gout = rets[1];

    // auto* y_end   = Y.buffer + Y.x.size();

    // Initialize array bounds
    auto* x_start = X.buffer;
    auto* x_end   = X.buffer + X.x.size();
    auto* y_start = Y.buffer;

    // Initialize the current position: 3 points
    auto* x_a = x_start;
    auto* x_b = x_a+1u;

    auto* y_a = y_start;
    auto* y_b = y_a+1u;

    auto* gout = Gout.buffer;
    auto* xout = Xout.buffer;
    while (x_b < x_end) {
        *xout = 0.5*(*x_b + *x_a);
        *gout = (*y_b-*y_a)/(*x_b - *x_a);

        ++x_a; ++x_b;
        ++y_a; ++y_b;
        ++xout; ++gout;
    }
}

