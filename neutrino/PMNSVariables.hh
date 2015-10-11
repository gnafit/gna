#ifndef PMNSVARIABLES_H
#define PMNSVARIABLES_H

#include <string>
#include <vector>
#include <complex>

#include <boost/format.hpp>

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
        allvars.add(&V[i][j], (boost::format("V%1%%2%") % i % j).str());
      }
    }
    return allvars;
  }
  void setExpressions(ExpressionsProvider &provider) {
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

class PMNSExpressions: public ExpressionsProvider {
public:
  PMNSExpressions()
    : ExpressionsProvider(new PMNSVariables(this))
    { }
};

#endif // PMNSVARIABLES_H
