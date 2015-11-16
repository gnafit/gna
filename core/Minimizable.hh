#ifndef MINIMIZABLE_H
#define MINIMIZABLE_H

#include "Math/IFunction.h"

#include "FitMeasure.hh"
#include "UncertainParameter.hh"

class Minimizable: public ROOT::Math::IMultiGenFunction {
public:
  Minimizable(FitMeasure *measure)
    : m_measure(measure)
  {
  }
public:
  void addParameter(Parameter<double> *par) {
    m_pars.push_back(par);
  }
  virtual ROOT::Math::IMultiGenFunction *Clone() const override {
    Minimizable *clone = new Minimizable(m_measure);
    for (auto par: m_pars) {
      clone->addParameter(par);
    }
    return clone;
  }
  virtual unsigned int NDim() const override {
    return m_pars.size();
  }
  virtual double DoEval(const double *x) const override {
    for (size_t i = 0; i < m_pars.size(); ++i) {
      m_pars[i]->set(x[i]);
    }
    return m_measure->value();
  }
protected:
  FitMeasure *m_measure;
  std::vector<Parameter<double>*> m_pars;
};

#endif // MINIMIZABLE_H
