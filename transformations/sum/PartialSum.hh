#pragma once
#include "GNAObject.hh"
#include "TransformationBind.hh"

class PartialSum: public GNAObject,
                 public TransformationBind<PartialSum> {
    public:
        using TransformationBind<PartialSum>::transformation_;
        PartialSum(double starting_value);
        
    private:
        void calc(FunctionArgs fargs);
        void findIdx(TypesFunctionArgs targs);

    double m_starting_value;
    size_t m_idx = 0;
};
