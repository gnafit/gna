#pragma once
#include "GNAObject.hh"
#include "TransformationBind.hh"

class PartialSum: public GNAObject,
                 public TransformationBind<PartialSum> {
    public:
        using TransformationBind<PartialSum>::transformation_;
        PartialSum();
        
    private:
        void calc(FunctionArgs fargs);
};
