#pragma once

#include <Eigen/Dense>

#include "OscProbPMNS.hh"
#include "Neutrino.hh"

class OscillationVariables;
class PMNSVariables;


namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscProbPMNSDecohT: public OscProbPMNSBaseT<FloatType>,
                            public TransformationBind<OscProbPMNSDecohT<FloatType>,FloatType,FloatType> {
    private:
      using BaseClass = OscProbPMNSBaseT<FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
      using TransformationBind<OscProbPMNSDecohT<FloatType>,FloatType,FloatType>::transformation_;
      OscProbPMNSDecohT(Neutrino from, Neutrino to);
      void calcSum(FunctionArgs fargs);
      template <int I, int J>
      void calcComponent(FunctionArgs fargs);
      template <int I, int J>
      void calcComponentCP(FunctionArgs fargs);
    protected:
      variable<double> m_L;
      variable<double> m_sigma;
    };
  }
}  
