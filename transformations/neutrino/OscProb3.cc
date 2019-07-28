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
        OscProb3T<FloatType>::OscProb3T(Neutrino from, Neutrino to, std::string l_name, bool modecos)
        : m_modecos(modecos), m_dm(3)
        {
            if (from.kind != to.kind) {
                throw std::runtime_error("particle-antiparticle oscillations");
            }
            m_alpha = from.flavor;
            m_beta = to.flavor;
            m_lepton_charge = from.leptonCharge();

            const char* dmnames[]  = { "DeltaMSq12", "DeltaMSq13", "DeltaMSq23" };
            for (int i = 0; i < 3; ++i) {
                variable_(&m_dm[i], dmnames[i]);
            }
            variable_(&m_L, l_name);

            add_transformation<0>();
            add_transformation<1>();
            add_transformation<2>();

            this->transformation_("compCP")
                .input("Enu")
                .output("compCP")
                .depends(m_L, m_dm[0], m_dm[1], m_dm[2])
                .types(new PassTypeT<FloatType>(0, {0,-1}))
                .func(&OscProb3::calcComponentCP)
#ifdef GNA_CUDA_SUPPORT
                .func("gpu", &OscProb3::gpuCalcComponentCP, DataLocation::Device)
                .storage("gpu", [](OscProb3::StorageTypesFunctionArgs& fargs){
                      fargs.ints[0] = DataType().points().shape(fargs.args[0].size());
                    })
#endif
                ;
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
                .func(m_modecos ? &OscProb3::calcComponent_modecos<I> : &OscProb3::calcComponent_modesin<I>)
#ifdef GNA_CUDA_SUPPORT
                .func("gpu", m_modecos ? &OscProb3::gpuCalcComponent_modecos<I> : &OscProb3::gpuCalcComponent_modesin<I>, DataLocation::Device)
                .storage("gpu", [](OscProb3::StorageTypesFunctionArgs& fargs){
                      fargs.ints[0] = DataType().points().shape(fargs.args[0].size());
                    })
#endif
                ;
        }

        template<typename FloatType>
        template <int I>
        void OscProb3T<FloatType>::calcComponent_modecos(typename OscProb3T<FloatType>::FunctionArgs& fargs) {
            auto &Enu = fargs.args[0].x;
            auto& ret = fargs.rets[0].x;
            ret = cos((m_dm[I].value()*oscprobArgumentFactor*m_L.value()*0.5)*Enu.inverse());
        }

        template<typename FloatType>
        template <int I>
        void OscProb3T<FloatType>::calcComponent_modesin(typename OscProb3T<FloatType>::FunctionArgs& fargs) {
            auto &Enu = fargs.args[0].x;
            auto& ret = fargs.rets[0].x;
            ret = sin((m_dm[I].value()*oscprobArgumentFactor*m_L.value()*0.25)*Enu.inverse()).square();
        }

        template<typename FloatType>
        void OscProb3T<FloatType>::calcComponentCP(typename OscProb3T<FloatType>::FunctionArgs& fargs) {
          auto& ret=fargs.rets[0].x;
          auto& Enu=fargs.args[0].x;
          auto tmp = (oscprobArgumentFactor*m_L.value()*0.25)*Enu.inverse().eval();
          ret = sin(m_dm[0].value()*tmp)
               *sin(m_dm[1].value()*tmp)
               *sin(m_dm[2].value()*tmp);
        }

#ifdef GNA_CUDA_SUPPORT
        template<typename FloatType>
        template<int I>
        void OscProb3T<FloatType>::gpuCalcComponent_modecos(typename OscProb3T<FloatType>::FunctionArgs& fargs) {
          fargs.args.touch();
          auto& gpuargs=fargs.gpu;
          gpuargs->provideSignatureDevice();
          cuCalcComponent_modecos<FloatType>(gpuargs->args, gpuargs->rets, gpuargs->ints, gpuargs->vars,
                                             fargs.args[0].arr.size(), gpuargs->nargs, oscprobArgumentFactor, m_dm[I].value(), m_L.value());
        }

        template<typename FloatType>
        template<int I>
        void OscProb3T<FloatType>::gpuCalcComponent_modesin(typename OscProb3T<FloatType>::FunctionArgs& fargs) {
          fargs.args.touch();
          auto& gpuargs=fargs.gpu;
          gpuargs->provideSignatureDevice();
          cuCalcComponent_modesin<FloatType>(gpuargs->args, gpuargs->rets, gpuargs->ints, gpuargs->vars,
                                             fargs.args[0].arr.size(), gpuargs->nargs, oscprobArgumentFactor, m_dm[I].value(), m_L.value());
        }

        template<typename FloatType>
        void OscProb3T<FloatType>::gpuCalcComponentCP(typename OscProb3T<FloatType>::FunctionArgs& fargs) {
          fargs.args.touch();
          auto& gpuargs=fargs.gpu;
          gpuargs->provideSignatureDevice();
          cuCalcComponentCP<FloatType>(gpuargs->args, gpuargs->rets, gpuargs->ints, gpuargs->vars, m_dm[0], m_dm[1], m_dm[2],
                                       fargs.args[0].arr.size(), gpuargs->nargs, oscprobArgumentFactor, m_L);
        }
#endif
    }
}

template class GNA::GNAObjectTemplates::OscProb3T<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::OscProb3T<float>;
#endif

