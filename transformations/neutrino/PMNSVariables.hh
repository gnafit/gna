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
  variable<double> V[Nnu][Nnu];
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
      .add(&V[0][0], {&Theta12, &Theta13}, [&](arrayview<double>& ret) {
          ret.complex() = cos(Theta12.value())*cos(Theta13.value());
        }, 2)
      .add(&V[0][1], {&Theta12, &Theta13}, [&](arrayview<double>& ret) {
          ret.complex() = sin(Theta12.value())*cos(Theta13.value());
        }, 2)
      .add(&V[0][2], {&Theta13, &Delta}, [&](arrayview<double>& ret) {
          auto phase = exp(-std::complex<double>(0, Delta.value()));
          ret.complex() = sin(Theta13.value())*phase;
        }, 2)
      .add(&V[1][0], {&Theta12, &Theta13, &Theta23, &Delta}, [&](arrayview<double>& ret) {
          auto phase = exp(std::complex<double>(0, Delta.value()));
          ret.complex() =
            -sin(Theta12.value())*cos(Theta23.value())
            -cos(Theta12.value())*sin(Theta23.value())*sin(Theta13.value())*phase;
        }, 2)
      .add(&V[1][1], {&Theta12, &Theta13, &Theta23, &Delta}, [&](arrayview<double>& ret) {
          auto phase = exp(std::complex<double>(0, Delta.value()));
          ret.complex() =
             cos(Theta12.value())*cos(Theta23.value())
            -sin(Theta12.value())*sin(Theta23.value())*sin(Theta13.value())*phase;
        }, 2)
      .add(&V[1][2], {&Theta13, &Theta23}, [&](arrayview<double>& ret) {
          ret.complex() = sin(Theta23.value())*cos(Theta13.value());
        }, 2)
      .add(&V[2][0], {&Theta12, &Theta13, &Theta23, &Delta}, [&](arrayview<double>& ret) {
          auto phase = exp(std::complex<double>(0, Delta.value()));
          ret.complex() =
            sin(Theta12.value())*sin(Theta23.value())
            -cos(Theta12.value())*cos(Theta23.value())*sin(Theta13.value())*phase;
        }, 2)
      .add(&V[2][1], {&Theta12, &Theta13, &Theta23, &Delta}, [&](arrayview<double>& ret) {
          auto phase = exp(std::complex<double>(0, Delta.value()));
          ret.complex() =
            -cos(Theta12.value())*sin(Theta23.value())
            -sin(Theta12.value())*cos(Theta23.value())*sin(Theta13.value())*phase;
        }, 2)
      .add(&V[2][2], {&Theta13, &Theta23}, [&](arrayview<double>& ret) {
          ret.complex() = cos(Theta23.value())*cos(Theta13.value());
        }, 2)
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
  variable<double> Phase;
  variable<double> PhaseC;
  variable<double> V[Nnu][Nnu];
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
      .add(&V[0][0], {&Cos12, &Cos13}, [&](arrayview<double>& ret) {
          ret.complex() = Cos12.value()*Cos13.value();
        }, 2)
      .add(&V[0][1], {&Sin12, &Cos13}, [&](arrayview<double>& ret) {
          ret.complex() = Sin12.value()*Cos13.value();
        }, 2)
      .add(&V[0][2], {&Sin13, &Phase}, [&](arrayview<double>& ret) {
          ret.complex() = Sin13.value()*Phase.complex();
        }, 2)
      .add(&V[1][0], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&](arrayview<double>& ret) {
          ret.complex() =
            -Sin12.value()*Cos23.value()
            -Cos12.value()*Sin23.value()*Sin13.value()*PhaseC.complex();
        }, 2)
      .add(&V[1][1], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&](arrayview<double>& ret) {
          ret.complex() =
             Cos12.value()*Cos23.value()
            -Sin12.value()*Sin23.value()*Sin13.value()*PhaseC.complex();
        }, 2)
      .add(&V[1][2], {&Cos13, &Sin23}, [&](arrayview<double>& ret) {
          ret.complex() = Sin23.value()*Cos13.value();
        }, 2)
      .add(&V[2][0], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&](arrayview<double>& ret) {
          ret.complex() =
            Sin12.value()*Sin23.value()
            -Cos12.value()*Cos23.value()*Sin13.value()*PhaseC.complex();
        }, 2)
      .add(&V[2][1], {&Sin12, &Cos12, &Sin13, &Sin23, &Cos23, &PhaseC}, [&](arrayview<double>& ret) {
          ret.complex() =
            -Cos12.value()*Sin23.value()
            -Sin12.value()*Cos23.value()*Sin13.value()*PhaseC.complex();
        }, 2)
      .add(&V[2][2], {&Cos13, &Cos23}, [&](arrayview<double>& ret) {
          ret.complex() = Cos23.value()*Cos13.value();
        }, 2)
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
