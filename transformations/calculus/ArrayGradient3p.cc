#include "ArrayGradient3p.hh"
#include "TypeClasses.hh"

#include <stdexcept>

ArrayGradient3p::ArrayGradient3p()
{
    using namespace TypeClasses;

    transformation_("gradient")
        .input("x")
        .input("y")
        .output("gradient")
        .types(new CheckSameTypesT<double>({0,-1}, "shape"), new PassTypeT<double>({1}, {0,0}))
        .types(new CheckNdimT<double>(1))
        .func(&ArrayGradient3p::calc_gradient);
}

OutputDescriptor ArrayGradient3p::inputs(SingleOutput& x, SingleOutput& y){
    const auto& t = t_[0];
    const auto& inputs = t.inputs();
    inputs[0].connect(x.single());
    inputs[1].connect(y.single());
    return OutputDescriptor(t.outputs()[0]);
}

void ArrayGradient3p::calc_gradient(FunctionArgs& fargs){
    // implements numpy.gradient technique
    auto& args = fargs.args;
    auto& Ret = fargs.rets[0];

    auto& X = args[0];
    auto& Y = args[1];

    // auto* y_end   = Y.buffer + Y.x.size();

    // Initialize array bounds
    auto* x_start = X.buffer;
    auto* x_end   = X.buffer + X.x.size();
    auto* y_start = Y.buffer;

    // Initialize the current position: 3 points
    auto* x_a = x_start;
    auto* x_b = x_a+1u;
    auto* x_c = x_b+1u;

    auto* y_a = y_start;
    auto* y_b = y_a+1u;
    auto* y_c = y_b+1u;

    auto* ret_b = Ret.buffer;

    auto h_s = *x_b - *x_a;
    auto h_d = *x_c - *x_b;

    // Calculate the left boundary
    *ret_b = (*y_b-*y_a)/h_s;

    // Loop over pairs
    ++ret_b;
    while (x_c < x_end) {
        if (fabs(h_d-h_s)<1e-14){
            *ret_b = 0.5*(*y_c-*y_a)/h_s;
        }
        else{
            auto h_s2 = h_s*h_s;
            auto h_d2 = h_d*h_d;
            *ret_b = ( h_s2*(*y_c) + (h_d2-h_s2)*(*y_b) - h_d2*(*y_a) )/( h_s*h_d*(h_s+h_d) );
        }

        // Make a step
        ++x_a; ++x_b; ++x_c;
        ++y_a; ++y_b; ++y_c;
        ++ret_b;
        h_s = h_d;
        h_d = *x_c - *x_b;
    }

    // Calculate the right boundary
    *ret_b = (*y_b-*y_a)/h_s;
}

