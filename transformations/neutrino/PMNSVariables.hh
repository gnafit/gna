#pragma once

#include <string>
#include <vector>
#include <complex>

#include "fmt/format.h"

#include "ParametersGroup.hh"

class PMNSVariables: public ParametersGroup {
public:
  static const size_t Nnu = 3;

  PMNSVariables(GNAObject *parent)
    : ParametersGroup(parent, fields())
    { }
  PMNSVariables(GNAObject *parent, std::vector<std::string> params)
    : PMNSVariables(parent)
    { initFields(params); }

  variable<double> Theta12;
  variable<double> Theta13;
  variable<double> Theta23;
  variable<double> Delta;
  variable<std::complex<double>> V[Nnu][Nnu];
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
      .add(&V[0][0], {&Theta12, &Theta13}, [&]() {
          return cos(Theta12)*cos(Theta13);
        })
      .add(&V[0][1], {&Theta12, &Theta13}, [&]() {
          return sin(Theta12)*cos(Theta13);
        })
      .add(&V[0][2], {&Theta13, &Delta}, [&]() {
          auto phase = exp(-std::complex<double>(0, Delta));
          return sin(Theta13)*phase;
        })
      .add(&V[1][0], {&Theta12, &Theta13, &Theta23, &Delta}, [&]() {
          auto phase = exp(std::complex<double>(0, Delta));
          return
            -sin(Theta12)*cos(Theta23)
            -cos(Theta12)*sin(Theta23)*sin(Theta13)*phase;
        })
      .add(&V[1][1], {&Theta12, &Theta13, &Theta23, &Delta}, [&]() {
          auto phase = exp(std::complex<double>(0, Delta));
          return
             cos(Theta12)*cos(Theta23)
            -sin(Theta12)*sin(Theta23)*sin(Theta13)*phase;
        })
      .add(&V[1][2], {&Theta13, &Theta23}, [&]() {
          return sin(Theta23)*cos(Theta13);
        })
      .add(&V[2][0], {&Theta12, &Theta13, &Theta23, &Delta}, [&]() {
          auto phase = exp(std::complex<double>(0, Delta));
          return
            sin(Theta12)*sin(Theta23)
            -cos(Theta12)*cos(Theta23)*sin(Theta13)*phase;
        })
      .add(&V[2][1], {&Theta12, &Theta13, &Theta23, &Delta}, [&]() {
          auto phase = exp(std::complex<double>(0, Delta));
          return
            -cos(Theta12)*sin(Theta23)
            -sin(Theta12)*cos(Theta23)*sin(Theta13)*phase;
        })
      .add(&V[2][2], {&Theta13, &Theta23}, [&]() {
          return cos(Theta23)*cos(Theta13);
        })
    ;
  }
};

class PMNSVariablesC: public ParametersGroup {
public:
  static const size_t Nnu = 3;

  PMNSVariablesC(GNAObject *parent)
    : ParametersGroup(parent, fields())
    { }
  PMNSVariablesC(GNAObject *parent, std::vector<std::string> params)
    : PMNSVariablesC(parent)
    { initFields(params); }

  variable<double> Sin12;
  variable<double> Sin13;
  variable<double> Sin23;
  variable<double> Cos12;
  variable<double> Cos13;
  variable<double> Cos23;
  variable<std::complex<double>> Phase;
  variable<std::complex<double>> PhaseC;
  variable<std::complex<double>> V[Nnu][Nnu];
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
      .add(&V[0][0], {&Cos12, &Cos13}, [&]() {
          return Cos12.value()*Cos13;
        })
      .add(&V[0][1], {&Sin12, &Cos13}, [&]() {
          return Sin12.value()*Cos13;
        })
      .add(&V[0][2], {&Sin13, &Phase}, [&]() {
          return Sin13.value()*Phase.value();
        })
      .add(&V[1][0], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&]() {
          return
            -Sin12.value()*Cos23
            -Cos12.value()*Sin23.value()*Sin13.value()*PhaseC.value();
        })
      .add(&V[1][1], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&]() {
          return
             Cos12.value()*Cos23
            -Sin12.value()*Sin23.value()*Sin13.value()*PhaseC.value();
        })
      .add(&V[1][2], {&Cos13, &Sin23}, [&]() {
          return Sin23.value()*Cos13;
        })
      .add(&V[2][0], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&]() {
          return
            Sin12.value()*Sin23
            -Cos12.value()*Cos23.value()*Sin13.value()*PhaseC.value();
        })
      .add(&V[2][1], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&]() {
          return
            -Cos12.value()*Sin23
            -Sin12.value()*Cos23.value()*Sin13.value()*PhaseC.value();
        })
      .add(&V[2][2], {&Cos13, &Cos23}, [&]() {
          return Cos23.value()*Cos13;
        })
    ;
  }
};

class PMNSExpressions: public ExpressionsProvider {
public:
  PMNSExpressions()
    : ExpressionsProvider(new PMNSVariables(this))
    { }
};

class PMNSExpressionsC: public ExpressionsProvider {
public:
  PMNSExpressionsC()
    : ExpressionsProvider(new PMNSVariablesC(this))
    { }
};
