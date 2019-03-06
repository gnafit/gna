#pragma once

#include <string>
#include <vector>
#include <complex>

#include "ParametersGroup.hh"

class OscillationVariables: public ParametersGroup {
public:
  OscillationVariables(GNAObject *parent)
    : ParametersGroup(parent, fields())
    { }
  OscillationVariables(GNAObject *parent, std::vector<std::string> params)
    : OscillationVariables(parent)
    { initFields(params); }

  variable<double> DeltaMSq12;
  variable<double> DeltaMSq13;
  variable<double> DeltaMSq23;
  variable<double> DeltaMSqEE;
  variable<double> DeltaMSqMM;
  variable<double> Alpha;
  variable<double> SinSq12;
  variable<double> SinSq13;
  variable<double> SinSq23;
  variable<double> CosSq12;
  variable<double> CosSq13;
  variable<double> CosSq23;
  variable<double> Sin12;
  variable<double> Sin13;
  variable<double> Sin23;
  variable<double> Cos12;
  variable<double> Cos13;
  variable<double> Cos23;
  variable<double> Theta12;
  variable<double> Theta13;
  variable<double> Theta23;
  variable<double> Delta;
  variable<std::complex<double>> Phase;
  variable<std::complex<double>> PhaseC;

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
      .add(&Phase, {&Delta}, [&]() {
          return exp(-std::complex<double>(0, Delta.value()));
        })
      .add(&PhaseC, {&Delta}, [&]() { //conjugate
          return exp(+std::complex<double>(0, Delta.value()));
        })
      ;
  }
};

class OscillationExpressions: public ExpressionsProvider {
public:
  OscillationExpressions()
    : ExpressionsProvider(new OscillationVariables(this))
    { }
};
