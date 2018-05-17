#pragma once

#include "GNAObject.hh"
#include "UncertainParameter.hh"
#include <vector>

class Jacobian: public GNAObject,
                public TransformationBind<Jacobian> {
public:
    Jacobian(double reldelta = 1e-1)
        : m_reldelta{reldelta}
    {
        transformation_(this, "jacobian")
            .input("func")
            .output("jacobian")
            .types(&Jacobian::calcTypes)
            .func(&Jacobian::calcJacobian)
            ;
    }
    
    void calcTypes(Atypes args, Rtypes rets);
    void calcJacobian(Args args, Rets rets);
    void dump();
    void append(Parameter<double>* par) {
        m_pars.push_back(par);
    };
protected:
    std::vector<Parameter<double>*> m_pars;
    double m_reldelta;
};
