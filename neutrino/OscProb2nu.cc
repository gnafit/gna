#include <TMath.h>

#include "OscProb2nu.hh"

using namespace Eigen;

static double km2MeV(double km) {
  return km*1E-3*TMath::Qe()/(TMath::Hbar()*TMath::C());
}

OscProb2nu::OscProb2nu()
  : m_param(new OscillationVariables(this))
{
  variable_(&m_L, "L");
  m_param->variable_("DeltaMSq12");
  m_param->variable_("SinSq12");
  transformation_(this, "prob")
    .input("Enu", DataType().points().any())
    .output("prob", DataType().points().any())
    .func([](OscProb2nu *obj, Args args, Rets rets) {
        obj->probability(args[0].x, rets[0].x);
      });
}

template <typename DerivedA, typename DerivedB>
void OscProb2nu::probability(const ArrayBase<DerivedA> &Enu,
                             ArrayBase<DerivedB> &ret) {
  auto w = 4.0*m_param->SinSq12*(1.0-m_param->SinSq12);
  ret = 1.0-w*sin(m_param->DeltaMSq12*km2MeV(m_L)/4.0/Enu).square();
}
