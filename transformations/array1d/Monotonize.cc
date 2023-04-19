#include "Monotonize.hh"
#include "TypeClasses.hh"

#include <stdexcept>

// #define DEBUG_MONOTONIZE
// #define DEBUG_ONLY_MONOTONIZE

Monotonize::Monotonize(double index_fraction, double gradient) :
m_index_fraction(index_fraction),
m_abs_gradient(std::fabs(gradient))
{
    using namespace TypeClasses;

    if(index_fraction<0 || index_fraction>=1){
        throw std::domain_error("Monotonize::index_fraction should be 0<=f<1");
    }
    transformation_("monotonize")
        .input("x")
        .input("y")
        .output("yout")
        .types(new CheckSameTypesT<double>({0,-1}, "shape"))
        .types(new CheckNdimT<double>(1))
        .types(&Monotonize::types)
        .func(&Monotonize::do_monotonize);
}

OutputDescriptor Monotonize::inputs(SingleOutput& x, SingleOutput& y){
    const auto& t = t_[0];
    const auto& inputs = t.inputs();
    inputs[0].connect(x.single());
    inputs[1].connect(y.single());
    return OutputDescriptor(t.outputs()[0]);
}

// OutputDescriptor Monotonize::inputs(SingleOutput& y){
//     const auto& t = t_[0];
//     const auto& inputs = t.inputs();
//     inputs[1].connect(y.single());
//     return OutputDescriptor(t.outputs()[0]);
// }

void Monotonize::types(TypesFunctionArgs& fargs){
    auto& args = fargs.args;

    m_has_x = args[0].defined();

    auto& y = args[1];
    if (!y.defined()){
        return;
    }
    fargs.rets[0] = y;

    m_index = static_cast<size_t>((y.shape[0]-1)*m_index_fraction);
}

void Monotonize::do_monotonize(FunctionArgs& fargs){
    auto& args = fargs.args;
    auto& Ret = fargs.rets[0];

    auto& Y = args[1];

    auto* y_start   = Y.buffer;
    auto* y_end     = Y.buffer + Y.x.size() - 1u;

    double* x_start = m_has_x ? args[0].buffer : nullptr;
    auto* x_current = x_start+m_index;

    auto* y_current = Y.buffer+m_index;
    auto* x_next    = x_current+1;
    auto* y_next    = y_current+1;

    // Setup return pointer and fill the current segment
    auto* ret_current = Ret.buffer+m_index;
    auto* ret_next    = ret_current+1;
    *ret_current = *y_current;
    *ret_next    = *y_next;

    // Define the step functions
    std::function<double()> get_x_step;
    std::function<void(int)> make_step;
    if(m_has_x){
        get_x_step = [&x_current, &x_next](){ return *x_next - *x_current; };
        make_step = [&x_current, &x_next, &y_current, &y_next, &ret_current, &ret_next](int dir){
            y_current+=dir; y_next+=dir;
            x_current+=dir; x_next+=dir;
            ret_current+=dir; ret_next+=dir;
        };
    }
    else{
        get_x_step = [](){ return 1.0; };
        make_step = [&y_current, &y_next, &ret_current, &ret_next](int dir){
            y_current+=dir; y_next+=dir;
            ret_current+=dir; ret_next+=dir;
        };
    }

    // Determine default direction, step
    double direction = *y_next>*y_current ? +1.0 : -1.0;
    make_step(+1);

    // Cycle forward
    while (y_current < y_end) {
        double direction_current = *y_next>*ret_current ? +1.0 : -1.0;

        if(direction_current==direction){
            *ret_next = *y_next;
        }
        else {
            auto step = get_x_step();
            *ret_next = *ret_current + direction*m_abs_gradient*step;
        }

#ifdef DEBUG_MONOTONIZE
#ifdef DEBUG_ONLY_MONOTONIZE
        if(direction_current!=direction){
#endif
            printf("step %4zu, dir %+.0f, cdir %+.0f, x %8.3f->%8.3f, y %8.3f->%8.3f, ret %8.3f->%8.3f\n",
                   size_t(y_current-y_start), direction, direction_current,
                   *x_current, *x_next,
                   *y_current, *y_next,
                   *ret_current, *ret_next);
#ifdef DEBUG_ONLY_MONOTONIZE
        }
#endif
#endif

        make_step(+1);
    }

    if(m_index==0){
        return;
    }

    x_next = x_start+m_index;
    y_next = Y.buffer+m_index;
    ret_next = Ret.buffer+m_index;
    x_current = x_next-1;
    y_current = y_next-1;
    ret_current = ret_next-1;
    while (y_current >= y_start) {
        double direction_current = *ret_next>*y_current ? +1.0 : -1.0;

        if(direction_current==direction){
            *ret_current = *y_current;
        }
        else{
            auto step = get_x_step();
            *ret_current = *ret_next - direction*m_abs_gradient*step;
        }

#ifdef DEBUG_MONOTONIZE
#ifdef DEBUG_ONLY_MONOTONIZE
        if(direction_current!=direction){
#endif
            printf("step %4zu, dir %+.0f, cdir %+.0f, x %8.3f->%8.3f, y %8.3f->%8.3f, ret %8.3f->%8.3f\n",
                   size_t(y_current-y_start), direction, direction_current,
                   *x_current, *x_next,
                   *y_current, *y_next,
                   *ret_current, *ret_next);
#ifdef DEBUG_ONLY_MONOTONIZE
        }
#endif
#endif

        make_step(-1);
    }

}

