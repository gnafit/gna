#pragma once

#include <Eigen/Dense>

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "config_vars.h"

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

class OscProbPMNSBase: public GNAObject,
                       public TransformationBind<OscProbPMNSBase> {
protected:
  OscProbPMNSBase(Neutrino from, Neutrino to);

  template <int I, int J>
  double DeltaMSq() const;

  template <int I, int J>
  double weight() const;

  double weightCP() const;

  std::unique_ptr<OscillationVariables> m_param;
  std::unique_ptr<PMNSVariables> m_pmns;

  int m_alpha, m_beta, m_lepton_charge;
};


namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscProbPMNST: public OscProbPMNSBase,
                        public TransformationBind<OscProbPMNST<FloatType>, FloatType, FloatType> {
    protected:
      using BaseClass = GNAObjectT<FloatType,FloatType>;
    public:
      using TransformationBind<OscProbPMNST<FloatType>, FloatType, FloatType>::transformation_;
      using typename BaseClass::FunctionArgs;

      OscProbPMNST<FloatType>(Neutrino from, Neutrino to, std::string l_name="L");

      template <int I, int J>
      void calcComponent(FunctionArgs& fargs);
      void calcComponentCP(FunctionArgs& fargs);
      void calcSum(FunctionArgs& fargs);
      void calcFullProb(FunctionArgs& fargs);
    #ifdef GNA_CUDA_SUPPORT
      void calcFullProbGpu(FunctionArgs& fargs);
      template <int I, int J>
      void gpuCalcComponent(FunctionArgs& fargs);
      void gpuCalcComponentCP(FunctionArgs& fargs);
      void gpuCalcSum(FunctionArgs& fargs);
    #endif

    protected:
      variable<FloatType> m_L;
    };
  }
}

using OscProbPMNS = GNA::GNAObjectTemplates::OscProbPMNST<double>;

class OscProbAveraged: public OscProbPMNSBase,
                       public TransformationBind<OscProbAveraged> {
public:
  using TransformationBind<OscProbAveraged>::transformation_;

  OscProbAveraged(Neutrino from, Neutrino to);
private:
  void CalcAverage(FunctionArgs fargs);
};

class OscProbPMNSMult: public OscProbPMNSBase,
                       public TransformationBind<OscProbPMNSMult> {
public:
  using TransformationBind<OscProbPMNSMult>::transformation_;

  OscProbPMNSMult(Neutrino from, Neutrino to, std::string l_name="Lavg");

  template <int I, int J>
  void calcComponent(FunctionArgs fargs);
  void calcSum(FunctionArgs fargs);
protected:
  variable<double> m_Lavg;

  variable<double> m_weights;
};
