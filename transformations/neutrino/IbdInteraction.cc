#include <TMath.h>

#include "IbdInteraction.hh"
#include "PDGVariables.hh"

#include "ParametricLazy.hpp"

IbdInteraction::IbdInteraction()
{
  m_pdg = new PDGVariables(this);
  m_pdg->variable_("ElectronMass");
  m_pdg->variable_("NeutronMass");
  m_pdg->variable_("ProtonMass");
  m_pdg->variable_("NeutronLifeTime");
  {
    using namespace ParametricLazyOps;
    m_threshold = mkdep(Pow(m_pdg->ElectronMass+m_pdg->NeutronMass, 2) -
                        Pow(m_pdg->ProtonMass, 2)/(2*m_pdg->ProtonMass));
    m_DeltaNP = mkdep(m_pdg->NeutronMass-m_pdg->ProtonMass);
    m_NucleonMass = mkdep((m_pdg->NeutronMass+m_pdg->ProtonMass)*0.5);
  }
  const double MeV2J = 1.e6*TMath::Qe();
  const double J2MeV = 1./MeV2J;
  const double MeVfactor = 1.E-6*TMath::Hbar()/TMath::Qe();
  m_cmfactor = MeVfactor * pow(TMath::Hbar()*TMath::C()*J2MeV, 2) * 1.E4;
}
