#pragma once

#include <iostream>
#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"

class SumAxis: public GNAObjectBind1N<double>,
               public TransformationBind<SumAxis> {
private:
    using GNAObjectT<double,double>::TypesFunctionArgs;
public:
    SumAxis(size_t axis, bool uninitialized=false);
    SumAxis(size_t axis, SingleOutput& output);

    TransformationDescriptor add_transformation(const std::string& name="");

protected:
    void SetTypes(TypesFunctionArgs& fargs);
    size_t m_axis;
};
