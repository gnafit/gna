#pragma once

#include <Eigen/Dense>

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "config_vars.h"

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscProb3T: public GNAObjectT<FloatType,FloatType>,
                     public TransformationBind<OscProb3<FloatType>,FloatType,FloatType> {
    protected:
      using BaseClass = GNAObjectT<FloatType,FloatType>;

    public:
      using TransformationBind<OscProbPMNST<FloatType>, FloatType, FloatType>::transformation_;
      using typename BaseClass::FunctionArgs;

      OscProb3(Neutrino from, Neutrino to, std::string l_name="L");

      protected:
        int m_alpha, m_beta, m_lepton_charge;
        variable<FloatType> m_L;
      };
  }
}


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

      //template <int I, int J>
      //void calcComponent(FunctionArgs& fargs);
      //void calcComponentCP(FunctionArgs& fargs);
      //void calcSum(FunctionArgs& fargs);
      //void calcFullProb(FunctionArgs& fargs);
    //#ifdef GNA_CUDA_SUPPORT
      //void calcFullProbGpu(FunctionArgs& fargs);
      //template <int I, int J>
      //void gpuCalcComponent(FunctionArgs& fargs);
      //void gpuCalcComponentCP(FunctionArgs& fargs);
      //void gpuCalcSum(FunctionArgs& fargs);
    //#endif
    };
  }
}

