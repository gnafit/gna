#pragma once

#include <Eigen/Dense>
#include "TypesFunctions.hh"
#include "TransformationFunctionArgs.hh"

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "config_vars.h"


class OscillationVariables;
class PMNSVariables;

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscProbPMNSBaseT: public GNAObjectT<FloatType,FloatType>,
                            public TransformationBind<OscProbPMNSBaseT<FloatType>, FloatType,FloatType> {

    private:
      using BaseClass = GNAObjectT<FloatType,FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
      using typename BaseClass::StorageTypesFunctionArgs;// = TransformationTypes::StorageTypesFunctionArgsT<FloatType,FloatType>;

    protected:
      OscProbPMNSBaseT(Neutrino from, Neutrino to);
        
      template <int I, int J>
      FloatType DeltaMSq() const {
        if (I == 1) {
          if (J == 2) return this->m_param->DeltaMSq12;
          if (J == 3) return this->m_param->DeltaMSq13;
        } else return this->m_param->DeltaMSq23;
      }
      
      template <int I, int J>
      FloatType weight() const {
        return std::real(
          this->m_pmns->V[m_alpha][I-1].value()*
          this->m_pmns->V[m_beta][J-1].value()*
          std::conj(this->m_pmns->V[m_alpha][J-1].value())*
          std::conj(this->m_pmns->V[m_beta][I-1].value())
          );
      }
        
      FloatType weightCP() const ;
        
      std::unique_ptr<OscillationVariables> m_param;
      std::unique_ptr<PMNSVariables> m_pmns;
        
      int m_alpha, m_beta, m_lepton_charge;
    };
  }    
}

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscProbPMNST: public OscProbPMNSBaseT<FloatType>,
                        public TransformationBind<OscProbPMNST<FloatType>, FloatType, FloatType> {

    private:
      using BaseClass = OscProbPMNSBaseT<FloatType>;
    public:
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
      using typename BaseClass::StorageTypesFunctionArgs;
      using TransformationBind<OscProbPMNST<FloatType>, FloatType, FloatType>::transformation_;
    
      OscProbPMNST<FloatType>(Neutrino from, Neutrino to, std::string l_name="L");
    
      template <int I, int J>
      void calcComponent(FunctionArgs fargs);
      void calcComponentCP(FunctionArgs fargs);
      void calcSum(FunctionArgs fargs);
      void calcFullProb(FunctionArgs fargs);
    #ifdef GNA_CUDA_SUPPORT
      void calcFullProbGpu(FunctionArgs fargs);
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


namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscProbAveragedT: public OscProbPMNSBaseT<FloatType>,
                           public TransformationBind<OscProbAveragedT<FloatType>, FloatType, FloatType> {
      using BaseClass =  OscProbPMNSBaseT<FloatType>;
    public:
      using TransformationBind<OscProbAveragedT<FloatType>, FloatType,FloatType>::transformation_;
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
    
      OscProbAveragedT<FloatType>(Neutrino from, Neutrino to);
    private:

      void CalcAverage(FunctionArgs fargs);
    };
    
    template<typename FloatType>
    class OscProbPMNSMultT: public OscProbPMNSBaseT<FloatType>,
                           public TransformationBind<OscProbPMNSMultT<FloatType>,FloatType,FloatType> {
      using BaseClass =  OscProbPMNSBaseT<FloatType>;
    public:
      using TransformationBind<OscProbPMNSMultT<FloatType>>::transformation_;
      using typename BaseClass::FunctionArgs;
      using typename BaseClass::TypesFunctionArgs;
    
      OscProbPMNSMultT<FloatType>(Neutrino from, Neutrino to, std::string l_name="Lavg");
    
      template <int I, int J>
      void calcComponent(FunctionArgs fargs);
      void calcSum(FunctionArgs fargs);
    protected:
      variable<double> m_Lavg;
    
      variable<double> m_weights;
    }; 
  }
}
