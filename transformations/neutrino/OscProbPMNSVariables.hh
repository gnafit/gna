#pragma once

#include <string>
#include <vector>
#include <complex>

#include <boost/format.hpp>

#include "ParametersGroup.hh"
#include "Neutrino.hh"

class OscProbPMNSVariables: public ParametersGroup {
public:
  static const size_t Nnu = 3;

  OscProbPMNSVariables(GNAObject *parent, Neutrino from, Neutrino to)
    : ParametersGroup(parent, fields(from, to))
    { }
  OscProbPMNSVariables(GNAObject *parent, std::vector<std::string> params, Neutrino from, Neutrino to)
    : OscProbPMNSVariables(parent, from, to)
    { initFields(params); }

  variable<std::complex<double>> V[Nnu][Nnu];

  variable<double> weight0;
  variable<double> weight12;
  variable<double> weight13;
  variable<double> weight23;
  variable<double> weightCP;
protected:

  Fields fields(Neutrino from, Neutrino to) {
    if (from.kind != to.kind) {
      throw std::runtime_error("particle-antiparticle oscillations");
    }
    m_alpha = from.flavor;
    m_beta  = to.flavor;
    m_lepton_charge = from.leptonCharge();

    Fields allvars;
    allvars
      .add(&weight0, "weight0")
      .add(&weight12, "weight12")
      .add(&weight13, "weight13")
      .add(&weight23, "weight23")
    ;
    if(m_alpha!=m_beta){
      allvars.add(&weightCP, "weightCP");
    }
    for (size_t i = 0; i < Nnu; ++i) {
      for (size_t j = 0; j < Nnu; ++j) {
        allvars.add(&V[i][j], (boost::format("V%1%%2%") % i % j).str());
      }
    }
    return allvars;
  }

  void setExpressions(ExpressionsProvider &provider) {
    provider
      .add(&weight0, {&weight12, &weight13, &weight23}, [&](){
           return static_cast<double>(m_alpha==m_beta)-(weight12.value()+weight13.value()+weight23.value());
           })
      .add(&weight12, {&V[m_alpha][0], &V[m_beta][0], &V[m_alpha][1], &V[m_beta][1]}, [&]() {
          return 2.0*std::real(
            V[m_alpha][0].value()*
            V[m_beta][1].value()*
            std::conj(V[m_alpha][1].value())*
            std::conj(V[m_beta][0].value())
            );
        })
      .add(&weight13, {&V[m_alpha][0], &V[m_beta][0], &V[m_alpha][2], &V[m_beta][2]}, [&]() {
          return 2.0*std::real(
            V[m_alpha][0].value()*
            V[m_beta][2].value()*
            std::conj(V[m_alpha][2].value())*
            std::conj(V[m_beta][0].value())
            );
        })
      .add(&weight23, {&V[m_alpha][1], &V[m_beta][1], &V[m_alpha][2], &V[m_beta][2]}, [&]() {
          return 2.0*std::real(
            V[m_alpha][1].value()*
            V[m_beta][2].value()*
            std::conj(V[m_alpha][2].value())*
            std::conj(V[m_beta][1].value())
            );
        })
      ;
      if(m_alpha!=m_beta){
        provider.add(&weightCP, {&V[m_alpha][0], &V[m_beta][0], &V[m_alpha][1], &V[m_beta][1]}, [&](){
          return m_lepton_charge*8.0*std::imag(
            V[m_alpha][0].value()*
            V[m_beta][1].value()*
            std::conj(V[m_alpha][1].value())*
            std::conj(V[m_beta][0].value())
            );
          });
      }
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

  int m_alpha, m_beta, m_lepton_charge;
};

class OscProbPMNSExpressions: public ExpressionsProvider {
public:
  OscProbPMNSExpressions(Neutrino from, Neutrino to)
    : ExpressionsProvider(new OscProbPMNSVariables(this, from, to))
    { }
};
