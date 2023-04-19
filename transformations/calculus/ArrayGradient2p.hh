#pragma once

#include "GNAObject.hh"
#include "boost/optional.hpp"

class ArrayGradient2p: public GNASingleObject,
                       public TransformationBind<ArrayGradient2p> {
public:
    ArrayGradient2p();
    ArrayGradient2p(SingleOutput& x, SingleOutput& y) : ArrayGradient2p() { inputs(x, y); }

    OutputDescriptor inputs(SingleOutput& x, SingleOutput& y);

private:
    void types(TypesFunctionArgs& fargs);
    void calc_gradient(FunctionArgs& fargs);
};
