#include "OscProb2nu.hh"

using namespace Eigen;

OscProb2nu::OscProb2nu()
  : m_param(new OscillationVariables(this))
{
  variable_(&m_L, "L");
  m_param->variable_("DeltaMSq12");
  m_param->variable_("SinSq12");
  transformation_("prob")
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
  ret = 1.0-w*sin(1.27*1e3*m_param->DeltaMSq12*m_L/Enu).square();
}
