#pragma once

#include <string>
#include <vector>
#include <complex>

#include "ParametersGroup.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class OscillationVariablesT: public GNA::GNAObjectTemplates::ParametersGroupT<FloatType> {
    public:
      using ParametersGroup = GNA::GNAObjectTemplates::ParametersGroupT<FloatType>;
      using ExpressionsProvider = GNA::GNAObjectTemplates::ExpressionsProviderT<FloatType>;
      using typename ParametersGroup::Fields;
      using ParametersGroup::initFields;

      OscillationVariablesT(GNAObjectT<FloatType,FloatType> *parent)
        : ParametersGroup(parent, fields())
        { }
      OscillationVariablesT(GNAObjectT<FloatType,FloatType> *parent, std::vector<std::string> params)
        : OscillationVariablesT(parent)
        { initFields(params); }

      variable<FloatType> DeltaMSq12;
      variable<FloatType> DeltaMSq13;
      variable<FloatType> DeltaMSq23;
      variable<FloatType> DeltaMSqEE;
      variable<FloatType> DeltaMSqMM;
      variable<FloatType> Alpha;
      variable<FloatType> SinSq12;
      variable<FloatType> SinSq13;
      variable<FloatType> SinSq23;
      variable<FloatType> CosSq12;
      variable<FloatType> CosSq13;
      variable<FloatType> CosSq23;
      variable<FloatType> Sin12;
      variable<FloatType> Sin13;
      variable<FloatType> Sin23;
      variable<FloatType> Cos12;
      variable<FloatType> Cos13;
      variable<FloatType> Cos23;
      variable<FloatType> Theta12;
      variable<FloatType> Theta13;
      variable<FloatType> Theta23;
      variable<FloatType> Delta;
      variable<FloatType> Phase;
      variable<FloatType> PhaseC;

    protected:
      Fields fields() {
        return Fields()
          .add(&DeltaMSq12, "DeltaMSq12")
          .add(&DeltaMSq13, "DeltaMSq13")
          .add(&DeltaMSq23, "DeltaMSq23")
          .add(&DeltaMSqEE, "DeltaMSqEE")
          .add(&DeltaMSqMM, "DeltaMSqMM")
          .add(&Alpha, "Alpha")
          .add(&SinSq12, "SinSq12")
          .add(&SinSq13, "SinSq13")
          .add(&SinSq23, "SinSq23")
          .add(&CosSq12, "CosSq12")
          .add(&CosSq13, "CosSq13")
          .add(&CosSq23, "CosSq23")
          .add(&Sin12, "Sin12")
          .add(&Sin13, "Sin13")
          .add(&Sin23, "Sin23")
          .add(&Cos12, "Cos12")
          .add(&Cos13, "Cos13")
          .add(&Cos23, "Cos23")
          .add(&Theta12, "Theta12")
          .add(&Theta13, "Theta13")
          .add(&Theta23, "Theta23")
          .add(&Delta, "Delta")
          .add(&Phase, "Phase")
          .add(&PhaseC, "PhaseC")
        ;
      }
      void setExpressions(ExpressionsProvider &provider) override {
        using std::sqrt;
        using std::asin;
        using std::sin;
        using std::pow;
        using std::exp;
        /* Syntax is the following: parameter to compute, {fields that are needed
         * for computation}, lambda that defines computation  */
        provider
          // Mass splittings
          .add(&DeltaMSq13,
               {&DeltaMSq23, &Alpha, &DeltaMSq12}, [&]() {
                 return DeltaMSq23.value() + Alpha.value()*DeltaMSq12.value();
               })
          .add(&DeltaMSq23,
               {&DeltaMSq13, &Alpha, &DeltaMSq12}, [&]() {
                 return DeltaMSq13.value() - Alpha.value()*DeltaMSq12.value();
               })
          .add(&DeltaMSq23,
               {&DeltaMSqEE, &Alpha, &SinSq12, &DeltaMSq12}, [&]() {
                 return DeltaMSqEE.value() + Alpha.value()*(SinSq12.value() - 1)*DeltaMSq12.value();
               })
          .add(&DeltaMSqEE,
               {&DeltaMSq23, &Alpha, &SinSq12, &DeltaMSq12}, [&]() {
                 return DeltaMSq23.value() - Alpha.value()*(SinSq12.value() - 1)*DeltaMSq12.value();
               })
          .add(&DeltaMSqMM, {&DeltaMSqEE, &Alpha, &Theta12, &Delta, &DeltaMSq12, &Theta23, &Theta13}, [&](){
                  return DeltaMSqEE.value() - Alpha.value()*sin(2*Theta12.value())
                         + cos(Delta.value())*sin(Theta13.value())*sin(2*Theta12.value())*tan(Theta23.value())*DeltaMSq12.value();})
          .add(&DeltaMSqEE, {&DeltaMSqMM, &Alpha, &Theta12, &Delta, &DeltaMSq12, &Theta23, &Theta13}, [&](){
                  return DeltaMSqMM.value() + Alpha.value()*sin(2*Theta12.value())
                         - cos(Delta.value())*sin(Theta13.value())*sin(2*Theta12.value())*tan(Theta23.value())*DeltaMSq12.value();})
          // Squared sine and cos
          .add(&SinSq12, {&Theta12}, [&]() { return pow(sin(Theta12.value()), 2); })
          .add(&SinSq13, {&Theta13}, [&]() { return pow(sin(Theta13.value()), 2); })
          .add(&SinSq23, {&Theta23}, [&]() { return pow(sin(Theta23.value()), 2); })
          .add(&CosSq12, {&SinSq12}, [&]() { return 1.0-SinSq12.value(); })
          .add(&CosSq13, {&SinSq13}, [&]() { return 1.0-SinSq13.value(); })
          .add(&CosSq23, {&SinSq23}, [&]() { return 1.0-SinSq23.value(); })
          // Sine and cos
          .add(&Sin12, {&SinSq12}, [&]() { return sqrt(SinSq12.value()); })
          .add(&Sin13, {&SinSq13}, [&]() { return sqrt(SinSq13.value()); })
          .add(&Sin23, {&SinSq23}, [&]() { return sqrt(SinSq23.value()); })
          .add(&Cos12, {&CosSq12}, [&]() { return sqrt(CosSq12.value()); })
          .add(&Cos13, {&CosSq13}, [&]() { return sqrt(CosSq13.value()); })
          .add(&Cos23, {&CosSq23}, [&]() { return sqrt(CosSq23.value()); })
          // Angles
          .add(&Theta12, {&Sin12}, [&]() { return asin(Sin12.value()); })
          .add(&Theta13, {&Sin13}, [&]() { return asin(Sin13.value()); })
          .add(&Theta23, {&Sin23}, [&]() { return asin(Sin23.value()); })
          // CP
          .add(&Phase, {&Delta}, [&](arrayview<FloatType>& ret) {
              ret.complex() = exp(-std::complex<FloatType>(0, Delta.value()));
            }, 2)
          .add(&PhaseC, {&Delta}, [&](arrayview<FloatType>& ret) { //conjugate
              ret.complex() = exp(+std::complex<FloatType>(0, Delta.value()));
            }, 2)
          ;
      }
    };

    template<typename FloatType>
    class OscillationExpressionsT: public ExpressionsProviderT<FloatType> {
    protected:
      using ExpressionsProvider = ExpressionsProviderT<FloatType>;
      using OscillationVariables = OscillationVariablesT<FloatType>;
    public:
      OscillationExpressionsT()
        : ExpressionsProvider(new OscillationVariables(this))
        { }
    };

  }
}

using OscillationExpressions = GNA::GNAObjectTemplates::OscillationExpressionsT<double>;
using OscillationVariables = GNA::GNAObjectTemplates::OscillationVariablesT<double>;

