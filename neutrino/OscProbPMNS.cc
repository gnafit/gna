#include <TMath.h>

#include "OscProbPMNS.hh"

#include "OscillationVariables.hh"
#include "PMNSVariables.hh"

using namespace Eigen;
using namespace Particles;

static double km2MeV(double km) {
  return km*1E-3*TMath::Qe()/(TMath::Hbar()*TMath::C());
}

OscProbPMNS::OscProbPMNS(Neutrino from, Neutrino to)
  : m_param(new OscillationVariables(this)), m_pmns(new PMNSVariables(this))
{
  m_alpha = from.flavor;
  m_beta = to.flavor;

  variable_(&m_L, "L");
  m_param->variable_("DeltaMSq12");
  m_param->variable_("DeltaMSq13");
  m_param->variable_("DeltaMSq23");
  for (size_t i = 0; i < m_pmns->Nnu; ++i) {
    m_pmns->variable_(&m_pmns->V[from.kind][i]);
    m_pmns->variable_(&m_pmns->V[to.kind][i]);
  }
  transformation_("comp12")
    .input("Enu", DataType().points().any())
    .output("comp", DataType().points().any())
    .depends(m_L, m_param->DeltaMSq12)
    .func(&OscProbPMNS::calcComponent<1,2>);
  transformation_("comp13")
    .input("Enu", DataType().points().any())
    .output("comp", DataType().points().any())
    .depends(m_L, m_param->DeltaMSq13)
    .func(&OscProbPMNS::calcComponent<1,3>);
  transformation_("comp23")
    .input("Enu", DataType().points().any())
    .output("comp", DataType().points().any())
    .depends(m_L, m_param->DeltaMSq23)
    .func(&OscProbPMNS::calcComponent<2,3>);
  if (from.flavor != to.flavor) {
    transformation_("compCP")
      .input("Enu", DataType().points().any())
      .output("comp", DataType().points().any())
      .depends(m_L)
      .depends(m_param->DeltaMSq12, m_param->DeltaMSq13, m_param->DeltaMSq23)
      .func(&OscProbPMNS::calcComponentCP);
  }
  auto probsum = transformation_("probsum")
    .input("", DataType().points().any())
    .input("comp12", DataType().points().any())
    .input("comp13", DataType().points().any())
    .input("comp23", DataType().points().any())
    .types(Atypes::pass<0>)
    .func(&OscProbPMNS::calcSum);
  if (from.flavor != to.flavor) {
    probsum.input("compJ", DataType().points().any());
  } else {
    probsum.input("compNoOscill", DataType().points().any());
  }
}

template <>
double OscProbPMNS::DeltaMSq<1,2>() const { return m_param->DeltaMSq12; }

template <>
double OscProbPMNS::DeltaMSq<1,3>() const { return m_param->DeltaMSq13; }

template <>
double OscProbPMNS::DeltaMSq<2,3>() const { return m_param->DeltaMSq23; }

template <int I, int J>
void OscProbPMNS::calcComponent(Args args, Rets rets) {
  auto &Enu = args[0].x;
  rets[0].x = sin(DeltaMSq<I,J>()*km2MeV(m_L)/4.0*Enu.inverse()).square();
}

void OscProbPMNS::calcComponentCP(Args args, Rets rets) {
  auto &Enu = args[0].x;
  ArrayXd tmp = km2MeV(m_L)/4.0*Enu.inverse();
  rets[0].x = sin(DeltaMSq<1,2>()*tmp);
  rets[0].x*= sin(DeltaMSq<1,3>()*tmp);
  rets[0].x*= sin(DeltaMSq<2,3>()*tmp);
}

template <int I, int J>
double OscProbPMNS::weight() const {
  return -4.0*real(
    m_pmns->V[m_alpha][I].value()*
    m_pmns->V[m_beta][J].value()*
    std::conj(m_pmns->V[m_alpha][J].value())*
    std::conj(m_pmns->V[m_beta][I].value())
  );
}

double OscProbPMNS::weightCP() const {
  return 8.0*imag(
    m_pmns->V[m_alpha][0].value()*
    m_pmns->V[m_beta][1].value()*
    std::conj(m_pmns->V[m_alpha][1].value())*
    std::conj(m_pmns->V[m_beta][0].value())
  );
}

void OscProbPMNS::calcSum(Args args, Rets rets) {
  rets[0].x = weight<1,2>()*args[0].x;
  rets[0].x+= weight<1,3>()*args[1].x;
  rets[0].x+= weight<2,3>()*args[2].x;
  if (m_alpha == m_beta) {
    rets[0].x += args[3].x;
  } else {
    rets[0].x += 8.0*weightCP()*args[3].x;
  }
}
