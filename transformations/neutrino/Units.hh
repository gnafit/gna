#pragma once
#include <array>
#include <boost/math/constants/constants.hpp>

namespace NeutrinoUnits {
    // elementary
    static constexpr double elementaryCharge_C = 1.602176634e-19;            // Qe,     Coulomb, PDG2020, exact
    static constexpr double speedOfLight_ms    = 299792458;                  // C,      m/s,     PDG2020, exact
    static constexpr double plankConstant_Js   = 6.62607015e-34;             // h,      J s,     PDG2020, exact
    static constexpr double hBar_Js            = plankConstant_Js*boost::math::double_constants::one_div_two_pi; // h/2pi,  J s
    static constexpr double hBar_eVs           = hBar_Js/elementaryCharge_C; // h/2pi,  eV s
    static constexpr double hBarC_eVm          = hBar_eVs*speedOfLight_ms;   // hc/2pi, MeV m
    static constexpr double N_A = 6.02214076e23; // Avogadro number, PDG2020, exact

    // energy
    static constexpr double MeV = 1.0;
    static constexpr double eV  = 1.e-6*MeV;
    static constexpr double keV = 1.e-3*MeV;
    static constexpr double GeV = 1.e3*MeV;
    static constexpr double TeV = 1.e6*MeV;

    static constexpr double Joule = eV/elementaryCharge_C;
    static constexpr double J     = Joule;
    static constexpr double kJ = 1.e3*J;
    static constexpr double MJ = 1.e6*J;
    static constexpr double GJ = 1.e9*J;
    static constexpr double TJ = 1.e12*J;

    // length
    static constexpr double m   =  1.0/hBarC_eVm/eV;
    static constexpr double nm  =  m*1.e-9;
    static constexpr double um  =  m*1.e-6;
    static constexpr double mm  =  m*1.e-3;
    static constexpr double cm  =  m*1.e-2;
    static constexpr double km  =  m*1.e3;

    // mass
    static constexpr double kg  = Joule*(speedOfLight_ms*speedOfLight_ms);
    static constexpr double g = 1e-3*kg;
    static constexpr double C_molar_mass = 11.9999999958*g; // carbon 12 molar mass, PDG2020
    static constexpr double aem = C_molar_mass/(12*N_A); // aem in MeV

    // area
    static constexpr double cm2 =  cm*cm;
    static constexpr double  m2 =   m*m;
    static constexpr double km2 =  km*km;

    // volume
    static constexpr double cm3 =  cm*cm*cm;

    // oscillation probability
    static constexpr double eV2 = eV*eV;
    static constexpr double oscprobArgumentFactor = km*eV2/MeV;

    // reactor power conversion: Convert W[GW]/[MeV] N[fissions] to N[fissions]/T[s]
    static constexpr double reactorPowerConversion = GJ/MeV;

    // elementary (natural)
    static constexpr double speedOfLight_MeVs = speedOfLight_ms/m;     // MeV/s

    // time TODO: make time consistent
    static constexpr double ns = 1.;
    static constexpr double year = ns * 1e9 * 60 * 60 * 24 * 365;

    //Fermi constant
    static constexpr double Gf = 1.1663787*1E-5/(GeV*GeV); // Gf/(hbar c)^3, PDG 2020
}
