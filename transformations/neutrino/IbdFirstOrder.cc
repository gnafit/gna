#include <boost/math/constants/constants.hpp>
#include <Eigen/Dense>

#include <TMath.h>

#include "IbdInteraction.hh"
#include "IbdFirstOrder.hh"
#include "PDGVariables.hh"
#include "TypesFunctions.hh"

#include "ParametricLazy.hpp"

const double pi = boost::math::constants::pi<double>();

using namespace Eigen;

IbdFirstOrder::IbdFirstOrder()
{
  transformation_("Enu")
    .input("Ee")
    .input("ctheta")
    .types([](Atypes args, Rtypes rets) {
        rets[0] = DataType().points().shape(args[0].size(), args[1].size());
      })
    .output("Enu")
    .func(&IbdFirstOrder::calc_Enu);
  transformation_("xsec")
    .input("Enu")
    .input("ctheta")
    .types(TypesFunctions::pass<0>)
    .output("xsec")
    .func(&IbdFirstOrder::calc_Xsec);
  transformation_("jacobian")
    .input("Enu")
    .input("Ee")
    .input("ctheta")
    .types(TypesFunctions::pass<0>)
    .output("jacobian")
    .func(&IbdFirstOrder::calc_dEnu_wrt_Ee);
  PDGVariables *p = m_pdg;
  {
    using namespace ParametricLazyOps;

    ElectronMass2 = mkdep(Pow(p->ElectronMass, 2));
    NeutronMass2 = mkdep(Pow(p->NeutronMass, 2));
    ProtonMass2 = mkdep(Pow(p->ProtonMass, 2));

    y2 = mkdep((Pow(m_DeltaNP, 2)-ElectronMass2)*0.5);
  }
}

void IbdFirstOrder::calc_Enu(FunctionArgs fargs) {
  const auto &Ee = fargs.args[0].x;
  const auto &ctheta = fargs.args[1].x;

  ArrayXd r = Ee / m_pdg->ProtonMass;
  ArrayXd Ve = (1.0 - ElectronMass2 / Ee.square()).sqrt();
  std::transform(Ve.data(), Ve.data() + Ve.size(), Ve.data(),
                [](double E){return (!std::isnan(E) ? E : 0.);});
  ArrayXd Ee0 = Ee + (NeutronMass2-ProtonMass2-ElectronMass2)/m_pdg->ProtonMass*0.5;
  ArrayXXd corr = (1.0-(1.0-(Ve.matrix()*ctheta.matrix().transpose()).array()).colwise()*r).inverse();
  fargs.rets[0].arr2d = corr.colwise()*Ee0;
}

double IbdFirstOrder::Xsec(double Eneu, double ctheta) {
  double Ee0 = Eneu - m_DeltaNP;
  if ( Ee0<=m_pdg->ElectronMass ) return 0.0;
  double pe0 = std::sqrt(Ee0*Ee0 - ElectronMass2);
  double ve0 = pe0 / Ee0;
  double ElectronMass5 = ElectronMass2 * ElectronMass2 * m_pdg->ElectronMass;
  double sigma0 = 2.* pi * pi /
    (PhaseFactor*(fsq+3.*gsq)*ElectronMass5*m_pdg->NeutronLifeTime/(1.E-6*TMath::Hbar()/TMath::Qe()));

  double Ee1 = Ee0 * ( 1.0 - Eneu/m_NucleonMass * ( 1.0 - ve0*ctheta ) ) - y2/m_NucleonMass;
  if (Ee1 <= m_pdg->ElectronMass) return 0.0;
  double pe1 = std::sqrt( Ee1*Ee1 - ElectronMass2 );
  double ve1 = pe1/Ee1;

  double sigma1a = sigma0*0.5 * ( ( fsq + 3.*gsq ) + ( fsq - gsq ) * ve1 * ctheta ) * Ee1 * pe1;

  double gamma_1 = 2.0 * g * ( f + f2 ) * ( ( 2.0 * Ee0 + m_DeltaNP ) * ( 1.0 - ve0 * ctheta ) - ElectronMass2/Ee0 );
  double gamma_2 = ( fsq + gsq ) * ( m_DeltaNP * ( 1.0 + ve0*ctheta ) + ElectronMass2/Ee0 );
  double A = ( ( Ee0 + m_DeltaNP ) * ( 1.0 - ctheta/ve0 ) - m_DeltaNP );
  double gamma_3 = ( fsq + 3.0*gsq )*A;
  double gamma_4 = ( fsq -     gsq )*A*ve0*ctheta;

  double sigma1b = -0.5 * sigma0 * Ee0 * pe0 * ( gamma_1 + gamma_2 + gamma_3 + gamma_4 ) / m_NucleonMass;

  return sigma1a + sigma1b;
}

void IbdFirstOrder::calc_Xsec(FunctionArgs fargs) {
  const auto &Eneu = fargs.args[0].arr2d;
  const auto &ctheta = fargs.args[1].x;
  auto &xsec = fargs.rets[0].arr2d;

  const double MeV2J = 1.E6 * TMath::Qe();
  const double J2MeV = 1./MeV2J;

  const double MeV2cm = pow(TMath::Hbar()*TMath::C()*J2MeV, 2) * 1.E4;

  for (int i = 0; i < Eneu.rows(); ++i) {
    for (int j = 0; j < Eneu.cols(); ++j) {
      xsec(i, j) = MeV2cm * Xsec(Eneu(i, j), ctheta(j));
    }
  }
}

void IbdFirstOrder::calc_dEnu_wrt_Ee(FunctionArgs fargs) {
  auto& args=fargs.args;
  const auto &Enu = args[0].arr2d;
  const auto &Ee = args[1].x;
  const auto &ctheta = args[2].x;
  auto& jacobian = fargs.rets[0].arr2d;

  ArrayXd Ve = (1.0 - ElectronMass2 / (Ee*Ee)).sqrt();
  ArrayXXd Vectheta = (Ve.matrix()*ctheta.matrix().transpose()).array();
  ArrayXXd ctheta_per_Ve = (Ve.inverse().matrix()*ctheta.matrix().transpose()).array();
  jacobian = (m_pdg->ProtonMass + (1.0-ctheta_per_Ve)*Enu )/( m_pdg->ProtonMass - (1-Vectheta).colwise()*Ee );
}
