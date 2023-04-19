#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "config_vars.h"

/**
 * @brief Approximated electron neutrino survival probability
 *
 * Based on Khan et al. [1910.12900]
 *
 * @author Maxim Gonchar
 * @date 2021.03
 */
class PsurEEMSWKhan: public GNAObject,
                     public TransformationBind<PsurEEMSWKhan> {
public:
    PsurEEMSWKhan(std::string l_name="L", std::string rho_name="rho", std::vector<std::string> dmnames={});
    PsurEEMSWKhan(double electron_fraction, std::string l_name="L", std::string rho_name="rho", std::vector<std::string> dmnames={});

protected:
    void calcOscProb(FunctionArgs& fargs);
    void types(TypesFunctionArgs& fargs);

    const double m_density_const;

    std::vector<variable<double>> m_dm;
    variable<double> m_dm_ee;
    variable<double> m_Alpha;
    variable<double> m_L;
    variable<double> m_rho;
    variable<double> m_cosSq13;
    variable<double> m_sinSq13;
    variable<double> m_sinSq12;
    variable<double> m_cosSq12;
    variable<double> m_cosDouble12;

    //
    // Common parts
    //
    Data<double>::ArrayType m_a;
    Data<double>::ArrayType m_common_21;
    Data<double>::ArrayType m_common_21_squared;
    Data<double>::ArrayType m_common_3;

    //
    // Modified oscillation parameterse
    //
    Data<double>::ArrayType m_sinSq13_mod;
    Data<double>::ArrayType m_sinSq12_mod;
    Data<double>::ArrayType m_dm21_mod;
    Data<double>::ArrayType m_dm31_mod;
    Data<double>::ArrayType m_dm32_mod;

};
