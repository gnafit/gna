#include <boost/math/constants/constants.hpp>
#include <Eigen/Dense>

#include <TMath.h>

#include "IbdInteraction.hh"
#include "IbdFirstOrder.hh"
#include "PDGVariables.hh"

#include "ParametricLazy.hpp"

const double pi = boost::math::constants::pi<double>();

using namespace Eigen;

IbdFirstOrder::IbdFirstOrder()
{
  transformation_("Enu")
    .input("Ee", DataType().points().any())
    .input("ctheta", DataType().points().any())
    .types([](Atypes args, Rtypes rets) {
        rets[0] = DataType().points().size(args[0].size*args[1].size);
        return Status::Success;
      })
    .output("Enu", DataType().points().any())
    .func(&IbdFirstOrder::calc_Enu);
  transformation_("xsec")
    .input("Enu", DataType().points().any())
    .input("ctheta", DataType().points().any())
    .types([](Atypes args, Rtypes rets) {
        rets[0] = args[0];
        return Status::Success;
      })
    .output("xsec", DataType().points().any())
    .func(&IbdFirstOrder::calc_Xsec);
  transformation_("jacobian")
    .input("Enu", DataType().points().any())
    .input("Ee", DataType().points().any())
    .input("ctheta", DataType().points().any())
    .types([](Atypes args, Rtypes rets) {
        rets[0] = args[0];
        return Status::Success;
      })
    .output("jacobian", DataType().points().any())
    .func(&IbdFirstOrder::calc_dEnu_wrt_Ee);
  PDGVariables *p = m_pdg;
  {
    using namespace ParametricLazyOps;

    ElectronMass2 = mkdep(Pow(p->ElectronMass, 2));
    NeutronMass2 = mkdep(Pow(p->NeutronMass, 2));
    ProtonMass2 = mkdep(Pow(p->ProtonMass, 2));

    DeltaNPE_tilded = mkdep((NeutronMass2-ProtonMass2-ElectronMass2)/p->ProtonMass*0.5);
    NeutronLifeTimeMeV = mkdep(p->NeutronLifeTime/(1.E-6*TMath::Hbar()/TMath::Qe()));
    y2 = mkdep((Pow(m_DeltaNP, 2)-ElectronMass2)*0.5);
  }
}

Status IbdFirstOrder::calc_Enu(Args args, Rets rets) {
  const auto &Ee = args[0].x;
  const auto &ctheta = args[1].x;

  ArrayXd r = Ee / m_pdg->ProtonMass;
  ArrayXd Ve = (1.0 - ElectronMass2 / (Ee*Ee)).sqrt();
  ArrayXd Ee0 = Ee + m_DeltaNP + (m_DeltaNP*m_DeltaNP - ElectronMass2)/(2*m_NucleonMass);
  ArrayXXd corr = 1.0/(1.0-(1.0-(Ve.matrix()*ctheta.matrix().transpose()).array()).colwise()*r);
  Map<ArrayXXd, Aligned>(rets[0].x.data(), Ee.size(), ctheta.size()) = corr.colwise()*Ee0;
  return Status::Success;
}

double IbdFirstOrder::Xsec(double Eneu, double ctheta) {
  double Ee0 = Eneu - m_DeltaNP;
  if ( Ee0<=m_pdg->ElectronMass ) return 0.0;
  double pe0 = std::sqrt(Ee0*Ee0 - ElectronMass2);
  double ve0 = pe0 / Ee0;
  double ElectronMass5 = ElectronMass2 * ElectronMass2 * m_pdg->ElectronMass;
  double sigma0 = 2.* pi * pi /
    (PhaseFactor*(fsq+3.*gsq)*ElectronMass5*m_pdg->NeutronLifeTime);

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

Status IbdFirstOrder::calc_Xsec(Args args, Rets rets) {
  std::cerr << "starting\n";
  size_t shape[2];
  shape[1] = args[1].x.size();
  std::cerr << "args[1]\n";
  shape[0] = args[0].x.size() / shape[1];
  std::cerr << "args[0]\n";
  Map<const ArrayXXd, Aligned> Eneu(args[0].x.data(), shape[0], shape[1]);
  const auto &ctheta = args[1].x;
  Map<ArrayXXd, Aligned> xsec(rets[0].x.data(), shape[0], shape[1]);

  std::cerr << "shape " << shape[0] << " " << shape[1] << "\n";
  std::cerr << "sizes " << Eneu.size() << " " << ctheta.size() << "\n";
  for (int i = 0; i < Eneu.size(); ++i) {
    for (int j = 0; j < ctheta.size(); ++j) {
      xsec(i, j) = Xsec(Eneu(i), ctheta(j));
    }
  }

  return Status::Success;
}

Status IbdFirstOrder::calc_dEnu_wrt_Ee(Args args, Rets rets) {
  size_t shape[2];
  shape[1] = args[2].x.size();
  shape[0] = args[1].x.size() / shape[1];
  Map<const ArrayXXd, Aligned> Enu(args[0].x.data(), shape[0], shape[1]);
  const auto &Ee = args[1].x;
  const auto &ctheta = args[2].x;
  Map<ArrayXXd, Aligned> jacobian(rets[0].x.data(), shape[0], shape[1]);

  ArrayXd Ve = (1.0 - ElectronMass2 / (Ee*Ee)).sqrt();
  ArrayXXd Vectheta = (Ve.matrix()*ctheta.matrix().transpose()).array();
  ArrayXXd Ve_per_ctheta = (Ve.inverse().matrix()*ctheta.matrix().transpose()).array();
  jacobian = (m_pdg->ProtonMass + (1.0-Ve_per_ctheta)*Enu )/( m_pdg->ProtonMass - (1-Vectheta).colwise()*Ee );

  return Status::Success;
}
