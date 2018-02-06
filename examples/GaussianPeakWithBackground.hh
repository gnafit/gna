#ifndef GAUSSIANPEAKWITHBACKGROUND_H
#define GAUSSIANPEAKWITHBACKGROUND_H

#include <boost/math/constants/constants.hpp>
#include <cmath>

#include "GNAObject.hh"
#include "ParametricLazy.hpp"

class GaussianPeakWithBackground: public GNAObject,
                                  public Transformation<GaussianPeakWithBackground> {
public:
  GaussianPeakWithBackground(double n=1) {
    variable_(&m_b, "BackgroundRate");
    variable_(&m_mu, "Mu");
    variable_(&m_E0, "E0");
    variable_(&m_w, "Width");
    using namespace ParametricLazyOps;

    m_w_scaled = mkdep(m_w*std::sqrt( n ));
    m_E0_scaled = mkdep(m_E0*n);

    transformation_(this, "rate")
      .input("E")
      .output("rate")
      .types(Atypes::pass<0,0>)
      .func(&GaussianPeakWithBackground::calcRate)
      ;
  }

  void calcRate(Args args, Rets rets) {
    const double pi = boost::math::constants::pi<double>();
    const auto &E = args[0].arr;
    rets[0].arr = m_b + m_mu*(1./std::sqrt(2*pi*m_w_scaled))*(-(E-m_E0_scaled).square()/(2*m_w_scaled*m_w_scaled)).exp();
  }
protected:
  variable<double> m_b, m_mu, m_E0, m_w;
  dependant<double> m_w_scaled, m_E0_scaled;
};

#endif // GAUSSIANPEAKWITHBACKGROUND_H
