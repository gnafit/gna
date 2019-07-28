#pragma once

#include <vector>
#include <boost/optional.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include "GNAObject.hh"

class HistEdgesLinear: public GNAObject,
                       public TransformationBind<HistEdgesLinear> {
public:
    HistEdgesLinear(double k, double b) : m_k(k), m_b(b) { init(); }
    HistEdgesLinear(SingleOutput& out, double k, double b) : HistEdgesLinear(k, b) { set(out); }

    void set(SingleOutput& out){ out.single() >> transformations.front().inputs.front(); }

private:
    void init();
    void types(TypesFunctionArgs& fargs);
    void func(FunctionArgs& fargs);

    double m_k;
    double m_b;

    DataType m_dt_hist_input;
    DataType m_dt_hist_output;
};
