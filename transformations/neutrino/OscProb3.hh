#pragma once

#include <vector>

#include "GNAObject.hh"
#include "Neutrino.hh"
#include "config_vars.h"
#undef GNA_CUDA_SUPPORT

namespace GNA {
    namespace GNAObjectTemplates {
        template<typename FloatType>
        class OscProb3T: public GNAObjectT<FloatType,FloatType>,
                         public TransformationBind<OscProb3T<FloatType>,FloatType,FloatType> {
            protected:
                using BaseClass = GNAObjectT<FloatType,FloatType>;
                using OscProb3  = OscProb3T<FloatType>;

            public:
                using TransformationBind<OscProb3T<FloatType>, FloatType, FloatType>::transformation_;
                using typename BaseClass::FunctionArgs;
                using typename BaseClass::StorageTypesFunctionArgs;
                using BaseClass::variable_;

                OscProb3T(Neutrino from, Neutrino to, std::string l_name="L");

                template<int I>
                void calcComponent(FunctionArgs& fargs);
                //void calcComponentCP(FunctionArgs& fargs);

                #ifdef GNA_CUDA_SUPPORT
                template<int I>
                void gpuCalcComponent(FunctionArgs& fargs);
                //void gpuCalcComponentCP(FunctionArgs& fargs);
                #endif

            protected:
                template<int I>
                void add_transformation();

                int m_alpha, m_beta, m_lepton_charge;
                std::vector<variable<FloatType>> m_dm;
                variable<FloatType> m_L;
            };
    }
}

