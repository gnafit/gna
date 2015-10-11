#ifndef OSCILLATIONVARIABLES_H
#define OSCILLATIONVARIABLES_H

#include <string>
#include <vector>

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
  variable<double> Delta;
  variable<double> Theta12;
  variable<double> Theta13;
  variable<double> Theta23;

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
      .add(&Delta, "Delta")
      .add(&Theta12, "Theta12")
      .add(&Theta13, "Theta13")
      .add(&Theta23, "Theta23")
    ;
  }
  virtual void setExpressions(ExpressionsProvider &provider) {
    using std::sqrt;
    using std::asin;
    using std::sin;
    using std::pow;

    provider
      .add(&DeltaMSq13,
           {&DeltaMSq23, &Alpha, &DeltaMSq12}, [&]() {
             return DeltaMSq23 + Alpha*DeltaMSq12;
           })
      .add(&DeltaMSq23,
           {&DeltaMSq13, &Alpha, &DeltaMSq12}, [&]() {
             return DeltaMSq13 - Alpha*DeltaMSq12;
           })
      .add(&DeltaMSq23,
           {&DeltaMSqEE, &Alpha, &SinSq12, &DeltaMSq12}, [&]() {
             return DeltaMSqEE + Alpha*(SinSq12 - 1)*DeltaMSq12;
           })
      .add(&DeltaMSqEE,
           {&DeltaMSq23, &Alpha, &SinSq12, &DeltaMSq12}, [&]() {
             return DeltaMSq23 - Alpha*(SinSq12 - 1)*DeltaMSq12;
           })
      .add(&Theta12, {&SinSq12}, [&]() { return asin(sqrt(SinSq12)); })
      .add(&SinSq12, {&Theta12}, [&]() { return pow(sin(Theta12), 2); })
      .add(&Theta13, {&SinSq13}, [&]() { return asin(sqrt(SinSq13)); })
      .add(&SinSq13, {&Theta13}, [&]() { return pow(sin(Theta13), 2); })
      .add(&Theta23, {&SinSq23}, [&]() { return asin(sqrt(SinSq23)); })
      .add(&SinSq23, {&Theta23}, [&]() { return pow(sin(Theta23), 2); })
      ;
  }
};

class OscillationExpressions: public ExpressionsProvider {
public:
  OscillationExpressions()
    : ExpressionsProvider(new OscillationVariables(this))
    { }
};

#endif // OSCILLATIONVARIABLES_H
