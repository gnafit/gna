#pragma once

#include <Eigen/Dense>

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "OscProbPMNS.hh"

class  OscProbMatter: public OscProbPMNSBase,
                      public TransformationBind<OscProbMatter> {
public:
    using TransformationBind<OscProbMatter>::transformation_;
    OscProbMatter(Neutrino from, Neutrino to, const std::string& baseline="L", const std::string& rho="rho");

    void calcOscProb(FunctionArgs fargs);

protected:
    variable<double> m_L;
    variable<double> m_rho;
    Neutrino m_from;
    Neutrino m_to;
};
