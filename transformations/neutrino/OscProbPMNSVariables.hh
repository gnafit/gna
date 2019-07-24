#pragma once

#include <string>
#include <vector>
#include <complex>

#include "fmt/format.h"

#include "ParametersGroup.hh"
#include "Neutrino.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscProbPMNSVariablesT: public ParametersGroupT<FloatType> {
    public:
      using ParametersGroup = GNA::GNAObjectTemplates::ParametersGroupT<FloatType>;
      using ExpressionsProvider = GNA::GNAObjectTemplates::ExpressionsProviderT<FloatType>;
      using typename ParametersGroup::Fields;
      using ParametersGroup::initFields;

      static const size_t Nnu = 3;

      OscProbPMNSVariablesT(GNAObjectT<FloatType,FloatType> *parent, Neutrino from, Neutrino to, const std::vector<std::string>& names={}, bool modecos=true)
        : ParametersGroup(parent, fields(from, to, names)), m_modecos(modecos)
        { }
      OscProbPMNSVariablesT(GNAObjectT<FloatType,FloatType> *parent, std::vector<std::string> params, Neutrino from, Neutrino to, const std::vector<std::string>& names={}, bool modecos=true)
        : OscProbPMNSVariablesT(parent, from, to, names, modecos)
        { initFields(params); }

      variable<FloatType> V[Nnu][Nnu];

      variable<FloatType> weight0;
      variable<FloatType> weight12;
      variable<FloatType> weight13;
      variable<FloatType> weight23;
      variable<FloatType> weightCP;
    protected:

      Fields fields(Neutrino from, Neutrino to, const std::vector<std::string>& names) {
        if (from.kind != to.kind) {
          throw std::runtime_error("particle-antiparticle oscillations");
        }
        m_alpha = from.flavor;
        m_beta  = to.flavor;
        m_lepton_charge = from.leptonCharge();

        std::vector<std::string> varnames;
        if(names.empty()){
          varnames={"weight0",
                    "weight12" , "weight13" , "weight23",
                    "weightCP",
          };
        }
        else if (names.size()==4u+static_cast<size_t>(m_alpha!=m_beta)){
          varnames=names;
        }
        else{
          throw std::runtime_error("Should provide 4(5) component names");
        }

        Fields allvars;
        allvars
          .add(&weight0,  varnames[0])
          .add(&weight12, varnames[1])
          .add(&weight13, varnames[2])
          .add(&weight23, varnames[3])
        ;
        if(m_alpha!=m_beta){
          allvars.add(&weightCP, varnames[4]);
        }
        for (size_t i = 0; i < Nnu; ++i) {
          for (size_t j = 0; j < Nnu; ++j) {
            allvars.add(&V[i][j], (fmt::format("V{0}{1}", i, j)));
          }
        }
        return allvars;
      }

      void setExpressions(ExpressionsProvider &provider) override {
        if(m_modecos){
          /// Mode A:
          /// P = Vak2 Vbk2 + 2 sum cos(.../2) + ...
          provider
            .add(&weight0, {&weight12, &weight13, &weight23}, [&](){
                 return static_cast<FloatType>(m_alpha==m_beta)-(weight12.value()+weight13.value()+weight23.value());
                 })
            .add(&weight12, {&V[m_alpha][0], &V[m_beta][0], &V[m_alpha][1], &V[m_beta][1]}, [&]() {
                return 2.0*std::real(
                  V[m_alpha][0].complex()*
                  V[m_beta][1].complex()*
                  std::conj(V[m_alpha][1].complex())*
                  std::conj(V[m_beta][0].complex())
                  );
              })
            .add(&weight13, {&V[m_alpha][0], &V[m_beta][0], &V[m_alpha][2], &V[m_beta][2]}, [&]() {
                return 2.0*std::real(
                  V[m_alpha][0].complex()*
                  V[m_beta][2].complex()*
                  std::conj(V[m_alpha][2].complex())*
                  std::conj(V[m_beta][0].complex())
                  );
              })
            .add(&weight23, {&V[m_alpha][1], &V[m_beta][1], &V[m_alpha][2], &V[m_beta][2]}, [&]() {
                return 2.0*std::real(
                  V[m_alpha][1].complex()*
                  V[m_beta][2].complex()*
                  std::conj(V[m_alpha][2].complex())*
                  std::conj(V[m_beta][1].complex())
                  );
              })
            ;
        }
        else{
          /// Mode B: delta + (1-cos) = delta + sin0.5
          /// Mode B:
          /// P = delta_ab - 4 sum sin²(.../4) + ...
            provider
              .add(&weight0, {&weight12, &weight13, &weight23}, [&](){return m_alpha==m_beta;})
              .add(&weight12, {&V[m_alpha][0], &V[m_beta][0], &V[m_alpha][1], &V[m_beta][1]}, [&]() {
                  return -4.0*std::real(
                    V[m_alpha][0].value()*
                    V[m_beta][1].value()*
                    std::conj(V[m_alpha][1].value())*
                    std::conj(V[m_beta][0].value())
                    );
                })
              .add(&weight13, {&V[m_alpha][0], &V[m_beta][0], &V[m_alpha][2], &V[m_beta][2]}, [&]() {
                  return -4.0*std::real(
                    V[m_alpha][0].value()*
                    V[m_beta][2].value()*
                    std::conj(V[m_alpha][2].value())*
                    std::conj(V[m_beta][0].value())
                    );
                })
              .add(&weight23, {&V[m_alpha][1], &V[m_beta][1], &V[m_alpha][2], &V[m_beta][2]}, [&]() {
                  return -4.0*std::real(
                    V[m_alpha][1].value()*
                    V[m_beta][2].value()*
                    std::conj(V[m_alpha][2].value())*
                    std::conj(V[m_beta][1].value())
                    );
                })
              ;
        }

        if(m_alpha!=m_beta){
          provider.add(&weightCP, {&V[m_alpha][0], &V[m_beta][0], &V[m_alpha][1], &V[m_beta][1]}, [&](){
            return m_lepton_charge*8.0*std::imag(
              V[m_alpha][0].value()*
              V[m_beta][1].value()*
              std::conj(V[m_alpha][1].value())*
              std::conj(V[m_beta][0].value())
              );
            });
          }
      }

      int m_alpha, m_beta, m_lepton_charge;
      bool m_modecos;
    };

    template<typename FloatType>
    class OscProbPMNSExpressionsT: public ExpressionsProviderT<FloatType> {
    public:
      OscProbPMNSExpressionsT(Neutrino from, Neutrino to, const std::vector<std::string>& names={}, bool modecos=true)
        : ExpressionsProviderT<FloatType>(new OscProbPMNSVariablesT<FloatType>(this, from, to, names, modecos))
        { }
    };
  }
}

using OscProbPMNSExpressions = GNA::GNAObjectTemplates::OscProbPMNSExpressionsT<double>;
using OscProbPMNSVariables = GNA::GNAObjectTemplates::OscProbPMNSVariablesT<double>;

