#pragma once

#include "Math/IFunction.h"

#include "Statistic.hh"
#include "UncertainParameter.hh"

#include <iostream>
#include <iomanip>

class MinimizableVerbose: public ROOT::Math::IMultiGenFunction {
public:
  MinimizableVerbose(Statistic *statistic) : m_statistic(statistic) { }

public:
  void addParameter(Parameter<double> *par) {
    m_pars.push_back(par);
  }

  ROOT::Math::IMultiGenFunction *Clone() const override {
    MinimizableVerbose *clone = new MinimizableVerbose(m_statistic);
    for (auto par: m_pars) {
      clone->addParameter(par);
    }
    return clone;
  }

  unsigned int NDim() const override {
    return m_pars.size();
  }

  double DoEval(const double *x) const override {
    using std::cerr;
    using std::endl;

    auto previous_precision = cerr.precision(12);
    for (size_t i = 0; i < m_pars.size(); ++i) {
      m_pars[i]->set(x[i]);
      std::cerr << m_pars[i]->qualifiedName() << ' ' << x[i] << std::endl;
    }
    double ret = m_statistic->value();
    std::cerr << "Statistic (" << m_ncall++ << "): " << ret << std::endl;
    std::cerr << std::endl;

    cerr.precision(previous_precision);
    return ret;
  }

protected:
  Statistic *m_statistic;
  std::vector<Parameter<double>*> m_pars;
  mutable size_t m_ncall{0u};
};
