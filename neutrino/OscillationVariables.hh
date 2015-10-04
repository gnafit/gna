#ifndef OSCILLATIONVARIABLES_H
#define OSCILLATIONVARIABLES_H

#include <string>
#include <vector>

#include "ParametersGroup.hh"

class OscillationVariables: public ParametersGroup {
public:
  OscillationVariables(GNAObject *parent)
    : ParametersGroup(parent, fields(), expressions())
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
    Fields allvars = {
      {"DeltaMSq12", &DeltaMSq12},
      {"DeltaMSq13", &DeltaMSq13},
      {"DeltaMSq23", &DeltaMSq23},
      {"DeltaMSqEE", &DeltaMSqEE},
      {"DeltaMSqMM", &DeltaMSqMM},
      {"Alpha", &Alpha},
      {"SinSq12", &SinSq12},
      {"SinSq13", &SinSq13},
      {"SinSq23", &SinSq23},
      {"Delta", &Delta},
      {"Theta12", &Theta12},
      {"Theta13", &Theta13},
      {"Theta23", &Theta23},
    };
    return allvars;
  }
  ExpressionsList expressions() {
    using std::sqrt;
    using std::asin;
    using std::sin;
    using std::pow;

    auto ret = ExpressionsList{
      {&DeltaMSq13, {&DeltaMSq23, &Alpha, &DeltaMSq12},
       [&]() {
         return DeltaMSq23 + Alpha*DeltaMSq12;
       }},
      {&DeltaMSq23, {&DeltaMSq13, &Alpha, &DeltaMSq12},
       [&]() {
         return DeltaMSq13 - Alpha*DeltaMSq12;
       }},
      {&DeltaMSq23, {&DeltaMSqEE, &Alpha, &SinSq12, &DeltaMSq12},
       [&]() {
         return DeltaMSqEE + Alpha*(SinSq12 - 1)*DeltaMSq12;
       }},
      {&DeltaMSqEE, {&DeltaMSq23, &Alpha, &SinSq12, &DeltaMSq12},
       [&]() {
         return DeltaMSq23 - Alpha*(SinSq12 - 1)*DeltaMSq12;
       }},
      {&Theta12, {&SinSq12}, [&]() { return asin(sqrt(SinSq12)); }},
      {&SinSq12, {&Theta12}, [&]() { return pow(sin(Theta12), 2); }},
      {&Theta13, {&SinSq13}, [&]() { return asin(sqrt(SinSq13)); }},
      {&SinSq13, {&Theta13}, [&]() { return pow(sin(Theta13), 2); }},
      {&Theta23, {&SinSq23}, [&]() { return asin(sqrt(SinSq23)); }},
      {&SinSq23, {&Theta23}, [&]() { return pow(sin(Theta23), 2); }},
    };
    return ret;
  }
};

class OscillationExpressions: public ExpressionsProvider {
public:
  OscillationExpressions()
    : ExpressionsProvider(new OscillationVariables(this))
    { }

  ClassDef(OscillationExpressions, 0);
};

#endif // OSCILLATIONVARIABLES_H
