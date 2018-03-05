#pragma once

#include "GNAObject.hh"
#include "UncertainParameter.hh"
#include <vector>
#include "Eigen/Dense"

class Jacobian: public GNAObject,
                public Transformation<Jacobian> {
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
        std::cout << "Append called" << std::endl;
        m_pars.push_back(par);
    };
protected:
    template <typename T>
    inline Eigen::ArrayXd computeDerivative(const T& input, Parameter<double>* x);
    std::vector<Parameter<double>*> m_pars;
    double m_reldelta;
};
