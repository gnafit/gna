#include "OscProb3.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

#ifdef GNA_CUDA_SUPPORT
#include "cuOscProbPMNS.hh"
#endif

#include "Units.hh"
using NeutrinoUnits::oscprobArgumentFactor;

namespace GNA {
    namespace GNAObjectTemplates {
        template<typename FloatType>
        OscProb3T<FloatType>::OscProb3T(Neutrino from, Neutrino to, std::string l_name)
        : m_dm(3)
        {
            if (from.kind != to.kind) {
                throw std::runtime_error("particle-antiparticle oscillations");
            }
            m_alpha = from.flavor;
            m_beta = to.flavor;
            if(m_alpha != m_beta){
                throw std::runtime_error("only survival probability is supported currently");
            }
            m_lepton_charge = from.leptonCharge();

            const char* dmnames[]  = { "DeltaMSq12", "DeltaMSq13", "DeltaMSq23" };
            for (int i = 0; i < 3; ++i) {
                variable_(&m_dm[i], dmnames[i]);
            }
            variable_(&m_L, l_name);

            add_transformation<0>();
            add_transformation<1>();
            add_transformation<2>();
        }

        template<typename FloatType>
        template <int I>
        void OscProb3T<FloatType>::add_transformation() {
            static const char* compnames[] = { "comp12", "comp13", "comp23" };
            this->transformation_(compnames[I])
                .input("Enu")
                .output(compnames[I])
                .depends(m_L, m_dm[I])
                .types(new PassTypeT<FloatType>(0, {0,-1}))
                .func(&OscProb3::calcComponent<I>)
#ifdef GNA_CUDA_SUPPORT
                .func("gpu", &OscProb3::gpuCalcComponent<I>, DataLocation::Device)
                .storage("gpu", [](OscProb3::StorageTypesFunctionArgs& fargs){
                      fargs.ints[0] = DataType().points().shape(fargs.args[0].size());
                    })
#endif
                ;
        }

        template<typename FloatType>
        template <int I>
        void OscProb3T<FloatType>::calcComponent(typename OscProb3T<FloatType>::FunctionArgs& fargs) {
            auto &Enu = fargs.args[0].x;
            auto& ret = fargs.rets[0].x;
            ret = cos((m_dm[I].value()*oscprobArgumentFactor*m_L.value()*0.5)*Enu.inverse());
        }

#ifdef GNA_CUDA_SUPPORT
        template<typename FloatType>
        template<int I>
        void OscProb3T<FloatType>::gpuCalcComponent(typename OscProb3T<FloatType>::FunctionArgs& fargs) {
          fargs.args.touch();
          auto& gpuargs=fargs.gpu;
          gpuargs->provideSignatureDevice();
          cuCalcComponent<FloatType>(gpuargs->args, gpuargs->rets, gpuargs->ints, gpuargs->vars,
                          fargs.args[0].arr.size(), gpuargs->nargs, oscprobArgumentFactor, m_dm[I].value(), m_L.value());
        }
#endif
    }
}

template class GNA::GNAObjectTemplates::OscProb3T<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::OscProb3T<float>;
#endif

