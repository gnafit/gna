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
      {"Delta", &Delta},
    };
    return allvars;
  }
  ExpressionsList expressions() {
    auto ret = ExpressionsList{
      {&DeltaMSq13, {&DeltaMSq23, &Alpha, &DeltaMSq12},
       [&]() -> double {
         return DeltaMSq23 + Alpha*DeltaMSq12;
       }},
      {&DeltaMSq23, {&DeltaMSq13, &Alpha, &DeltaMSq12},
       [&]() -> double {
         return DeltaMSq13 - Alpha*DeltaMSq12;
       }},
      {&DeltaMSq23, {&DeltaMSqEE, &Alpha, &SinSq12, &DeltaMSq12},
       [&]() -> double {
         return DeltaMSqEE + Alpha*(SinSq12 - 1)*DeltaMSq12;
       }},
      {&DeltaMSqEE, {&DeltaMSq23, &Alpha, &SinSq12, &DeltaMSq12},
       [&]() -> double {
         return DeltaMSq23 - Alpha*(SinSq12 - 1)*DeltaMSq12;
       }},
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
