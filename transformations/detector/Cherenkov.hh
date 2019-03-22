#pragma once
#include "GNAObject.hh"
#include "TransformationBind.hh"

class Cherenkov: public GNAObject,
                 public TransformationBind<Cherenkov> {
    public:
        using TransformationBind<Cherenkov>::transformation_;
        Cherenkov();
        
    private:
        void calc_Cherenkov(FunctionArgs fargs);
        variable<double> p0, p1, p2, p3, p4;
        variable<double> E_0;
};
