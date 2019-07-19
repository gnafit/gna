#pragma once

#include <Eigen/Dense>

#include "OscProbPMNS.hh"
#include "Neutrino.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscillationVariablesT;

    template<typename FloatType>
    class PMNSVariablesT;
  }
}
using OscillationVariables = GNA::GNAObjectTemplates::OscillationVariablesT<double>;
using PMNSVariables = GNA::GNAObjectTemplates::PMNSVariablesT<double>;

class OscProbPMNSDecoh: public OscProbPMNSBase,
                        public TransformationBind<OscProbPMNSDecoh> {
public:
  using TransformationBind<OscProbPMNSDecoh>::transformation_;
  OscProbPMNSDecoh(Neutrino from, Neutrino to);
  void calcSum(FunctionArgs fargs);
  template <int I, int J>
  void calcComponent(FunctionArgs fargs);
  template <int I, int J>
  void calcComponentCP(FunctionArgs fargs);
protected:
  variable<double> m_L;
  variable<double> m_sigma;
};

