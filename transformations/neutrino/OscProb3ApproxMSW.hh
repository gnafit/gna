#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "config_vars.h"

/**
 * @brief Approximated electron neutrino survival probability
 *
 * Based on JUNO Yellow Book
 *
 * @author Konstantin Treskov
 */
class OscProb3ApproxMSW: public GNAObject,
                         public TransformationBind<OscProb3ApproxMSW> {
public:
    OscProb3ApproxMSW(std::string l_name="L",std::string rho_name="rho", std::vector<std::string> dmnames={});

    void calcOscProb(FunctionArgs& fargs);

protected:
    int m_alpha, m_beta;
    std::vector<variable<double>> m_dm;
    variable<double> m_L;
    variable<double> m_rho;
    variable<double> m_sinSqDouble13;
    variable<double> m_sinSqDouble12;
    variable<double> m_cosDouble12;
    variable<double> m_cosSq13;
};
