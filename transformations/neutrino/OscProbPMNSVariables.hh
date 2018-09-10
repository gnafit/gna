#pragma once

#include <string>
#include <vector>
#include <complex>

#include <boost/format.hpp>

#include "ParametersGroup.hh"

class OscProbPMNSVariables: public ParametersGroup {
public:
  static const size_t Nnu = 3;

  OscProbPMNSVariables(GNAObject *parent)
    : ParametersGroup(parent, fields())
    { }
  OscProbPMNSVariables(GNAObject *parent, std::vector<std::string> params)
    : OscProbPMNSVariables(parent)
    { initFields(params); }

  variable<std::complex<double>> V[Nnu][Nnu];
  variable<double> test;
protected:
  Fields fields() {
    Fields allvars;
    allvars
      .add(&test, "test")
    ;
    for (size_t i = 0; i < Nnu; ++i) {
      for (size_t j = 0; j < Nnu; ++j) {
        allvars.add(&V[i][j], (boost::format("V%1%%2%") % i % j).str());
      }
    }
    return allvars;
  }
  void setExpressions(ExpressionsProvider &provider) {
    //using std::sin;
    //using std::cos;
    //using std::exp;
    provider
      .add(&test, {&V[0][0]}, [&]() {
          return std::real(V[0][0].value())-10.0;
        })
    ;
      //.add(&V[0][1], {&Theta12, &Theta13}, [&]() {
          //return sin(Theta12)*cos(Theta13);
        //})
      //.add(&V[0][2], {&Theta13, &Delta}, [&]() {
          //auto phase = exp(-std::complex<double>(0, Delta));
          //return sin(Theta13)*phase;
        //})
      //.add(&V[1][0], {&Theta12, &Theta13, &Theta23, &Delta}, [&]() {
          //auto phase = exp(std::complex<double>(0, Delta));
          //return
            //-sin(Theta12)*cos(Theta23)
            //-cos(Theta12)*sin(Theta23)*sin(Theta13)*phase;
        //})
      //.add(&V[1][1], {&Theta12, &Theta13, &Theta23, &Delta}, [&]() {
          //auto phase = exp(std::complex<double>(0, Delta));
          //return
             //cos(Theta12)*cos(Theta23)
            //-sin(Theta12)*sin(Theta23)*sin(Theta13)*phase;
        //})
      //.add(&V[1][2], {&Theta13, &Theta23}, [&]() {
          //return sin(Theta23)*cos(Theta13);
        //})
      //.add(&V[2][0], {&Theta12, &Theta13, &Theta23, &Delta}, [&]() {
          //auto phase = exp(std::complex<double>(0, Delta));
          //return
            //sin(Theta12)*sin(Theta23)
            //-cos(Theta12)*cos(Theta23)*sin(Theta13)*phase;
        //})
      //.add(&V[2][1], {&Theta12, &Theta13, &Theta23, &Delta}, [&]() {
          //auto phase = exp(std::complex<double>(0, Delta));
          //return
            //-cos(Theta12)*sin(Theta23)
            //-sin(Theta12)*cos(Theta23)*sin(Theta13)*phase;
        //})
      //.add(&V[2][2], {&Theta13, &Theta23}, [&]() {
          //return cos(Theta23)*cos(Theta13);
        //})
    //;
  }
};

class OscProbPMNSExpressions: public ExpressionsProvider {
public:
  OscProbPMNSExpressions()
    : ExpressionsProvider(new OscProbPMNSVariables(this))
    { }
};
