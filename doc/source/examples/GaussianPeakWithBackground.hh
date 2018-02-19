#ifndef GAUSSIANPEAKWITHBACKGROUND_H
#define GAUSSIANPEAKWITHBACKGROUND_H

#include <boost/math/constants/constants.hpp>
#include <cmath>

#include "GNAObject.hh"

class GaussianPeakWithBackground: public GNAObject,
                                  public TransformationBlock<GaussianPeakWithBackground> {
public:
  GaussianPeakWithBackground() {
    variable_(&m_b, "BackgroundRate");
    variable_(&m_mu, "Mu");
    variable_(&m_E0, "E0");
    variable_(&m_w, "Width");

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
    rets[0].arr = m_b + m_mu*(1./std::sqrt(2*pi*m_w))*(-(E-m_E0).square()/(2*m_w*m_w)).exp();
  }
protected:
  variable<double> m_b, m_mu, m_E0, m_w;
};

#endif // GAUSSIANPEAKWITHBACKGROUND_H
