#pragma once
#include "GNAObject.hh"
#include "TransformationBind.hh"

class Cherenkov_Borexino: public GNAObject,
                 public TransformationBind<Cherenkov_Borexino> {
    public:
        using TransformationBind<Cherenkov_Borexino>::transformation_;
        Cherenkov_Borexino();
        
    private:
        void calc_Cherenkov(FunctionArgs fargs);
        variable<double> p0, p1, p2, p3, p4;
        variable<double> E_0;
};
