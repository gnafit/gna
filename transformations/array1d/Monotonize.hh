#pragma once

#include "GNAObject.hh"
#include "boost/optional.hpp"

class Monotonize: public GNASingleObject,
                  public TransformationBind<Monotonize> {
public:
    Monotonize(double index_fraction=0, double gradient=0);
    // Monotonize(SingleOutput& y, double index_fraction=0, double gradient=0) : Monotonize(index_fraction, gradient) { inputs(y); }
    Monotonize(SingleOutput& x, SingleOutput& y, double index_fraction=0, double gradient=0) : Monotonize(index_fraction, gradient) { inputs(x, y); }

    // OutputDescriptor inputs(SingleOutput& y);
    OutputDescriptor inputs(SingleOutput& x, SingleOutput& y);

private:
    void types(TypesFunctionArgs& fargs);
    void do_monotonize(FunctionArgs& fargs);

    double m_index_fraction{0.0};
    double m_abs_gradient{0.0};

    size_t m_index{0};
    bool m_has_x{false};
};
