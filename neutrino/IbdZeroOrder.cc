#include <boost/math/constants/constants.hpp>

#include <TMath.h>

#include "IbdInteraction.hh"
#include "IbdZeroOrder.hh"
#include "PDGVariables.hh"

#include <iostream>

const double pi = boost::math::constants::pi<double>();

IbdZeroOrder::IbdZeroOrder()
{
  transformation_(this, "Enu")
    .input("Ee", DataType().points().any())
    .output("Enu", DataType().points().any())
    .func(&IbdZeroOrder::calcEnu);
  transformation_(this, "xsec")
    .input("Ee", DataType().points().any())
    .output("xsec", DataType().points().any())
    .func(&IbdZeroOrder::calcXsec);
}

void IbdZeroOrder::calcEnu(Args args, Rets rets) {
  rets[0].x = args[0].x + m_DeltaNP;
}

void IbdZeroOrder::calcXsec(Args args, Rets rets) {
  const auto &Ee = args[0].x;

  const double MeV2J = 1.E6 * TMath::Qe();
  const double J2MeV = 1./MeV2J;

  const double MeV2cm = pow(TMath::Hbar()*TMath::C()*J2MeV, 2) * 1.E4;

  auto pe = (Ee.square() - m_pdg->ElectronMass*m_pdg->ElectronMass).sqrt();
  auto coeff = 2.*pi*pi /
    (std::pow(m_pdg->ElectronMass, 5) * PhaseFactor * m_pdg->NeutronLifeTime/(1.E-6*TMath::Hbar()/TMath::Qe()));
  rets[0].x = MeV2cm * coeff*Ee*pe;
}
