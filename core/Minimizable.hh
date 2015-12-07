#ifndef MINIMIZABLE_H
#define MINIMIZABLE_H

#include "Math/IFunction.h"

#include "Statistic.hh"
#include "UncertainParameter.hh"

#include <iostream>

class Minimizable: public ROOT::Math::IMultiGenFunction {
public:
  Minimizable(Statistic *statistic)
    : m_statistic(statistic)
  {
  }
public:
  void addParameter(Parameter<double> *par) {
    m_pars.push_back(par);
  }
  ROOT::Math::IMultiGenFunction *Clone() const override {
    Minimizable *clone = new Minimizable(m_statistic);
    for (auto par: m_pars) {
      clone->addParameter(par);
    }
    return clone;
  }
  unsigned int NDim() const override {
    return m_pars.size();
  }
  double DoEval(const double *x) const override {
    for (size_t i = 0; i < m_pars.size(); ++i) {
      m_pars[i]->set(x[i]);
    }
    double ret = m_statistic->value();
    return ret;
  }
protected:
  Statistic *m_statistic;
  std::vector<Parameter<double>*> m_pars;
};

#endif // MINIMIZABLE_H
