#pragma once

#include <string>
#include <vector>
#include <complex>

#include "fmt/format.h"

#include "ParametersGroup.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class PMNSVariablesT: public ParametersGroupT<FloatType> {
    public:
      using GNAObject = GNAObjectT<FloatType,FloatType>;
      using ParametersGroup = GNA::GNAObjectTemplates::ParametersGroupT<FloatType>;
      using ExpressionsProvider = GNA::GNAObjectTemplates::ExpressionsProviderT<FloatType>;
      using typename ParametersGroup::Fields;
      using ParametersGroup::initFields;

      static const size_t Nnu = 3;

      PMNSVariablesT(GNAObject *parent)
        : ParametersGroup(parent, fields())
        { }
      PMNSVariablesT(GNAObject *parent, std::vector<std::string> params)
        : PMNSVariablesT(parent)
        { initFields(params); }

      variable<FloatType> Theta12;
      variable<FloatType> Theta13;
      variable<FloatType> Theta23;
      variable<FloatType> Delta;
      variable<FloatType> V[Nnu][Nnu];
    protected:
      Fields fields() {
        Fields allvars;
        allvars
          .add(&Theta12, "Theta12")
          .add(&Theta13, "Theta13")
          .add(&Theta23, "Theta23")
          .add(&Delta, "Delta")
        ;
        for (size_t i = 0; i < Nnu; ++i) {
          for (size_t j = 0; j < Nnu; ++j) {
            allvars.add(&V[i][j], fmt::format("V{0}{1}", i, j));
          }
        }
        return allvars;
      }
      void setExpressions(ExpressionsProvider &provider) override {
        using std::sin;
        using std::cos;
        using std::exp;
        provider
          .add(&V[0][0], {&Theta12, &Theta13}, [&](arrayview<FloatType>& ret) {
              ret.complex() = cos(Theta12.value())*cos(Theta13.value());
            }, 2)
          .add(&V[0][1], {&Theta12, &Theta13}, [&](arrayview<FloatType>& ret) {
              ret.complex() = sin(Theta12.value())*cos(Theta13.value());
            }, 2)
          .add(&V[0][2], {&Theta13, &Delta}, [&](arrayview<FloatType>& ret) {
              auto phase = exp(-std::complex<FloatType>(0, Delta.value()));
              ret.complex() = sin(Theta13.value())*phase;
            }, 2)
          .add(&V[1][0], {&Theta12, &Theta13, &Theta23, &Delta}, [&](arrayview<FloatType>& ret) {
              auto phase = exp(std::complex<FloatType>(0, Delta.value()));
              ret.complex() =
                -sin(Theta12.value())*cos(Theta23.value())
                -cos(Theta12.value())*sin(Theta23.value())*sin(Theta13.value())*phase;
            }, 2)
          .add(&V[1][1], {&Theta12, &Theta13, &Theta23, &Delta}, [&](arrayview<FloatType>& ret) {
              auto phase = exp(std::complex<FloatType>(0, Delta.value()));
              ret.complex() =
                 cos(Theta12.value())*cos(Theta23.value())
                -sin(Theta12.value())*sin(Theta23.value())*sin(Theta13.value())*phase;
            }, 2)
          .add(&V[1][2], {&Theta13, &Theta23}, [&](arrayview<FloatType>& ret) {
              ret.complex() = sin(Theta23.value())*cos(Theta13.value());
            }, 2)
          .add(&V[2][0], {&Theta12, &Theta13, &Theta23, &Delta}, [&](arrayview<FloatType>& ret) {
              auto phase = exp(std::complex<FloatType>(0, Delta.value()));
              ret.complex() =
                sin(Theta12.value())*sin(Theta23.value())
                -cos(Theta12.value())*cos(Theta23.value())*sin(Theta13.value())*phase;
            }, 2)
          .add(&V[2][1], {&Theta12, &Theta13, &Theta23, &Delta}, [&](arrayview<FloatType>& ret) {
              auto phase = exp(std::complex<FloatType>(0, Delta.value()));
              ret.complex() =
                -cos(Theta12.value())*sin(Theta23.value())
                -sin(Theta12.value())*cos(Theta23.value())*sin(Theta13.value())*phase;
            }, 2)
          .add(&V[2][2], {&Theta13, &Theta23}, [&](arrayview<FloatType>& ret) {
              ret.complex() = cos(Theta23.value())*cos(Theta13.value());
            }, 2)
        ;
      }
    };

    template<typename FloatType>
    class PMNSVariablesCT: public ParametersGroupT<FloatType> {
    public:
      using GNAObject = GNAObjectT<FloatType,FloatType>;
      using ParametersGroup = GNA::GNAObjectTemplates::ParametersGroupT<FloatType>;
      using ExpressionsProvider = GNA::GNAObjectTemplates::ExpressionsProviderT<FloatType>;
      using typename ParametersGroup::Fields;
      using ParametersGroup::initFields;

      static const size_t Nnu = 3;

      PMNSVariablesCT(GNAObject *parent)
        : ParametersGroup(parent, fields())
        { }
      PMNSVariablesCT(GNAObject *parent, std::vector<std::string> params)
        : PMNSVariablesCT(parent)
        { initFields(params); }

      variable<FloatType> Sin12;
      variable<FloatType> Sin13;
      variable<FloatType> Sin23;
      variable<FloatType> Cos12;
      variable<FloatType> Cos13;
      variable<FloatType> Cos23;
      variable<FloatType> Phase;
      variable<FloatType> PhaseC;
      variable<FloatType> V[Nnu][Nnu];
    protected:
      Fields fields() {
        Fields allvars;
        allvars
          .add(&Sin12, "Sin12") .add(&Sin13, "Sin13") .add(&Sin23, "Sin23")
          .add(&Cos12, "Cos12") .add(&Cos13, "Cos13") .add(&Cos23, "Cos23")
          .add(&Phase, "Phase")
          .add(&PhaseC, "PhaseC")
        ;
        for (size_t i = 0; i < Nnu; ++i) {
          for (size_t j = 0; j < Nnu; ++j) {
            allvars.add(&V[i][j], fmt::format("V{0}{1}", i, j));
          }
        }
        return allvars;
      }
      void setExpressions(ExpressionsProvider &provider) override {
        using std::exp;
        provider
          .add(&V[0][0], {&Cos12, &Cos13}, [&](arrayview<FloatType>& ret) {
              ret.complex() = Cos12.value()*Cos13.value();
            }, 2)
          .add(&V[0][1], {&Sin12, &Cos13}, [&](arrayview<FloatType>& ret) {
              ret.complex() = Sin12.value()*Cos13.value();
            }, 2)
          .add(&V[0][2], {&Sin13, &Phase}, [&](arrayview<FloatType>& ret) {
              ret.complex() = Sin13.value()*Phase.complex();
            }, 2)
          .add(&V[1][0], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&](arrayview<FloatType>& ret) {
              ret.complex() =
                -Sin12.value()*Cos23.value()
                -Cos12.value()*Sin23.value()*Sin13.value()*PhaseC.complex();
            }, 2)
          .add(&V[1][1], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&](arrayview<FloatType>& ret) {
              ret.complex() =
                 Cos12.value()*Cos23.value()
                -Sin12.value()*Sin23.value()*Sin13.value()*PhaseC.complex();
            }, 2)
          .add(&V[1][2], {&Cos13, &Sin23}, [&](arrayview<FloatType>& ret) {
              ret.complex() = Sin23.value()*Cos13.value();
            }, 2)
          .add(&V[2][0], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&](arrayview<FloatType>& ret) {
              ret.complex() =
                Sin12.value()*Sin23.value()
                -Cos12.value()*Cos23.value()*Sin13.value()*PhaseC.complex();
            }, 2)
          .add(&V[2][1], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&](arrayview<FloatType>& ret) {
              ret.complex() =
                -Cos12.value()*Sin23.value()
                -Sin12.value()*Cos23.value()*Sin13.value()*PhaseC.complex();
            }, 2)
          .add(&V[2][2], {&Cos13, &Cos23}, [&](arrayview<FloatType>& ret) {
              ret.complex() = Cos23.value()*Cos13.value();
            }, 2)
        ;
      }
    };

    template<typename FloatType>
    class PMNSExpressionsT: public ExpressionsProviderT<FloatType> {
    public:
      PMNSExpressionsT()
        : ExpressionsProviderT<FloatType>(new PMNSVariablesT<FloatType>(this))
        { }
    };

    template<typename FloatType>
    class PMNSExpressionsCT: public ExpressionsProviderT<FloatType> {
    public:
      PMNSExpressionsCT()
        : ExpressionsProviderT<FloatType>(new PMNSVariablesCT<FloatType>(this))
        { }
    };
  }
}

using PMNSExpressions = GNA::GNAObjectTemplates::PMNSExpressionsT<double>;
using PMNSVariables = GNA::GNAObjectTemplates::PMNSVariablesT<double>;
