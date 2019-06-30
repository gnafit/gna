#pragma once

#include <Eigen/Dense>

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "OscProbPMNS.hh"



namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class  OscProbMatterT: public OscProbPMNSBaseT<FloatType>,
                          public TransformationBind<OscProbMatterT<FloatType>,FloatType,FloatType> {
    private:
        using BaseClass = OscProbPMNSBaseT<FloatType>;
    public:
        using typename BaseClass::FunctionArgs;
        using typename BaseClass::TypesFunctionArgs;
        using TransformationBind<OscProbMatterT<FloatType>,FloatType,FloatType>::transformation_;
        OscProbMatterT(Neutrino from, Neutrino to);
    
        void calcOscProb(FunctionArgs fargs);
    
    protected:
        variable<double> m_L;
        variable<double> m_rho;
        Neutrino m_from;
        Neutrino m_to;
    };
  }
}
