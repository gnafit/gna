#pragma once

#include "GNAObject.hh"
#include "boost/optional.hpp"

class ArrayGradient3p: public GNASingleObject,
                       public TransformationBind<ArrayGradient3p> {
public:
    ArrayGradient3p();
    ArrayGradient3p(SingleOutput& x, SingleOutput& y) : ArrayGradient3p() { inputs(x, y); }

    OutputDescriptor inputs(SingleOutput& x, SingleOutput& y);

private:
    void calc_gradient(FunctionArgs& fargs);
};
