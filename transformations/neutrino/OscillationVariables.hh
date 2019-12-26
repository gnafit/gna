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

      // Mass splittings
      variable<FloatType> DeltaMSq12;
      variable<FloatType> DeltaMSq13;
      variable<FloatType> DeltaMSq23;
      // Effective mass splittings
      variable<FloatType> DeltaMSqEE;
      variable<FloatType> DeltaMSqMM;
      // Neutrino mass ordering
      variable<FloatType> Alpha;
      // Sines (squared) of mixing agles
      variable<FloatType> SinSq12;
      variable<FloatType> SinSq13;
      variable<FloatType> SinSq23;
      // Cosines of double mixing agles
      variable<FloatType> CosDouble12;
      variable<FloatType> CosDouble13;
      // Sines (squared) of double mixing agles
      variable<FloatType> SinSqDouble12;
      variable<FloatType> SinSqDouble13;
      // Cosines (squared) of mixing agles
      variable<FloatType> CosSq12;
      variable<FloatType> CosSq13;
      variable<FloatType> CosSq23;
      // Sines of mixing agles
      variable<FloatType> Sin12;
      variable<FloatType> Sin13;
      variable<FloatType> Sin23;
      // Cosines of mixing agles
      variable<FloatType> Cos12;
      variable<FloatType> Cos13;
      variable<FloatType> Cos23;
      // Mixing agles
      variable<FloatType> Theta12;
      variable<FloatType> Theta13;
      variable<FloatType> Theta23;
      // CP violation
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
          .add(&CosDouble12, "CosDouble12")
          .add(&CosDouble13, "CosDouble13")
          .add(&SinSqDouble12, "SinSqDouble12")
          .add(&SinSqDouble13, "SinSqDouble13")
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
               }, "Δm²(13)=Δm²(23) + αΔm²(12)")
          .add(&DeltaMSq23,
               {&DeltaMSq13, &Alpha, &DeltaMSq12}, [&]() {
                 return DeltaMSq13.value() - Alpha.value()*DeltaMSq12.value();
               }, "Δm²(23)=Δm²(13) - αΔm²(12)")
          .add(&DeltaMSq23,
               {&DeltaMSqEE, &Alpha, &SinSq12, &DeltaMSq12}, [&]() {
                 return DeltaMSqEE.value() + Alpha.value()*(SinSq12.value() - 1)*DeltaMSq12.value();
               }, "Δm²(23)=Δm²(ee) + α w(12) Δm²(12)")
          .add(&DeltaMSqEE,
               {&DeltaMSq23, &Alpha, &SinSq12, &DeltaMSq12}, [&]() {
                 return DeltaMSq23.value() - Alpha.value()*(SinSq12.value() - 1)*DeltaMSq12.value();
               }, "Δm²(ee) =Δm²(23) - α w(12) Δm²(12)")
          .add(&DeltaMSqMM, {&DeltaMSqEE, &Alpha, &Theta12, &Delta, &DeltaMSq12, &Theta23, &Theta13}, [&](){
                  return DeltaMSqEE.value() - Alpha.value()*sin(2*Theta12.value())
                         + cos(Delta.value())*sin(Theta13.value())*sin(2*Theta12.value())*tan(Theta23.value())*DeltaMSq12.value();},
               "Δm²(mm) = f(Δm²(ee), ...)" )
          .add(&DeltaMSqEE, {&DeltaMSqMM, &Alpha, &Theta12, &Delta, &DeltaMSq12, &Theta23, &Theta13}, [&](){
                  return DeltaMSqMM.value() + Alpha.value()*sin(2*Theta12.value())
                         - cos(Delta.value())*sin(Theta13.value())*sin(2*Theta12.value())*tan(Theta23.value())*DeltaMSq12.value();},
              "Δm²(ee) = f(Δm²(μμ), ...)")
          // Squared sine and squared cos from cos of double angle
          .add(&SinSq12, {&CosDouble12}, [&]() { return 0.5*(1.0-CosDouble12.value()); }, "sin²θ(12)=(1-cos2θ(12))/2")
          .add(&SinSq13, {&CosDouble13}, [&]() { return 0.5*(1.0-CosDouble13.value()); }, "sin²θ(13)=(1-cos2θ(13))/2")
          .add(&CosSq12, {&CosDouble12}, [&]() { return 0.5*(1.0+CosDouble12.value()); }, "cos²θ(12)=(1+cos2θ(12))/2")
          .add(&CosSq13, {&CosDouble13}, [&]() { return 0.5*(1.0+CosDouble13.value()); }, "cos²θ(13)=(1+cos2θ(13))/2")
          // Squared sine from angle
          .add(&SinSq12, {&Theta12}, [&]() { return pow(sin(Theta12.value()), 2); }, "sin²θ(12)")
          .add(&SinSq13, {&Theta13}, [&]() { return pow(sin(Theta13.value()), 2); }, "sin²θ(13)")
          .add(&SinSq23, {&Theta23}, [&]() { return pow(sin(Theta23.value()), 2); }, "sin²θ(23)")
          // Squared sine from squared cos
          .add(&SinSq12, {&CosSq12}, [&]() { return 1.0-CosSq12.value(); }, "cos²θ(12)=1-sin²θ(12)")
          .add(&SinSq13, {&CosSq13}, [&]() { return 1.0-CosSq13.value(); }, "cos²θ(13)=1-sin²θ(13)")
          .add(&SinSq23, {&CosSq23}, [&]() { return 1.0-CosSq23.value(); }, "cos²θ(23)=1-sin²θ(23)")
          // Sine and cos
          .add(&Sin12, {&SinSq12}, [&]() { return sqrt(SinSq12.value()); }, "sinθ(12)=sqrt sin²θ(12)")
          .add(&Sin13, {&SinSq13}, [&]() { return sqrt(SinSq13.value()); }, "sinθ(13)=sqrt sin²θ(13)")
          .add(&Sin23, {&SinSq23}, [&]() { return sqrt(SinSq23.value()); }, "sinθ(23)=sqrt sin²θ(23)")
          .add(&Cos12, {&CosSq12}, [&]() { return sqrt(CosSq12.value()); }, "cosθ(12)=sqrt cos²θ(12)")
          .add(&Cos13, {&CosSq13}, [&]() { return sqrt(CosSq13.value()); }, "cosθ(13)=sqrt cos²θ(13)")
          .add(&Cos23, {&CosSq23}, [&]() { return sqrt(CosSq23.value()); }, "cosθ(23)=sqrt cos²θ(23)")
          // Angles
          .add(&Theta12, {&Sin12}, [&]() { return asin(Sin12.value()); }, "θ(12)=asin sinθ(12)")
          .add(&Theta13, {&Sin13}, [&]() { return asin(Sin13.value()); }, "θ(13)=asin sinθ(13)")
          .add(&Theta23, {&Sin23}, [&]() { return asin(Sin23.value()); }, "θ(23)=asin sinθ(23)")
          // CP
          .add(&Phase, {&Delta}, [&](arrayview<FloatType>& ret) {
              ret.complex() = exp(-std::complex<FloatType>(0, Delta.value()));
            }, 2, "-exp(iδ)")
          .add(&PhaseC, {&Delta}, [&](arrayview<FloatType>& ret) { //conjugate
              ret.complex() = exp(+std::complex<FloatType>(0, Delta.value()));
            }, 2, "+exp(iδ)")
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

