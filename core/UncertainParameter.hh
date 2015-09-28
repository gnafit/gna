#ifndef UNCERTAINPARAMETER_H
#define UNCERTAINPARAMETER_H

#include <string>
#include <limits>
#include <utility>

#include <TObject.h>

#include "Parameters.hh"

class ParameterWrapper: public TObject {
public:
  ParameterWrapper(const std::string &name)
    : m_var(name.c_str()) { }
  ParameterWrapper(parameter<double> &pvar)
    : m_var(pvar) { }

  double value() const { return m_var; }
  void set(double value) { m_var = value; }

  const variable<double> &getVariable() const { return m_var; }
protected:
  parameter<double> m_var;

  ClassDef(ParameterWrapper, 0);
};

class Uncertain: public TObject {
public:
  Uncertain(const std::string &name)
    : m_name(name) { }
  virtual ~Uncertain() { }

  const std::string &name() const { return m_name; }
  virtual double value() = 0;
  virtual double central() = 0;
  virtual void setCentral(double value) = 0;
  virtual double sigma() = 0;
  virtual void setSigma(double sigma) = 0;
protected:
  std::string m_name;

  ClassDef(Uncertain, 0);
};

typedef std::pair<double, double> LimitsPair;

class Parameter: public Uncertain {
public:
  Parameter(const std::string &name)
    : Uncertain(name) { }

  virtual void set(double value) = 0;
  virtual double relativeValue(double diff) = 0;
  virtual void relativeShift(double diff) { set(relativeValue(diff)); }

  virtual double cast(const std::string &v) const;
  virtual double cast(const double &v) const { return v; }

  virtual void addLimits(double min, double max)
    { m_limits.push_back(std::make_pair(min, max)); }
  virtual const std::vector<LimitsPair> &limits() const
    { return m_limits; }

  virtual void reset() { set(central()); }
protected:
  std::vector<LimitsPair> m_limits;

  ClassDef(Parameter, 0);
};

class GaussianParameter: public Parameter {
public:
  GaussianParameter(const std::string &name)
    : Parameter(name), m_var(name.c_str()) { }
  GaussianParameter(const std::string &name, parameter<double> pvar)
    : Parameter(name), m_var(pvar) { }

  double value() { return m_var;}
  double central() { return m_central; }
  void setCentral(double value) { m_central = value; }
  double sigma() { return m_sigma; }
  void setSigma(double sigma) { m_sigma = sigma; }
  void set(double value) { m_var = value;}
  double relativeValue(double diff) { return value() + diff*m_sigma; }

  const variable<double> &getVariable() const { return m_var; }
protected:
  parameter<double> m_var;
  double m_central;
  double m_sigma;

  ClassDef(GaussianParameter, 0);
};

class GaussianValue: public Uncertain {
public:
  GaussianValue(const std::string &name, variable<double> var)
    : Uncertain(name), m_var(var) { }

  double value() { return m_var; }
  double central() { return m_central; }
  void setCentral(double value) { m_central = value; }
  double sigma() { return m_sigma; }
  void setSigma(double sigma) { m_sigma = sigma; }

  const variable<double> &getVariable() const { return m_var; }
protected:
  variable<double> m_var;
  double m_central;
  double m_sigma;

  ClassDef(GaussianValue, 0);
};

class UniformAngleParameter: public Parameter {
public:
  UniformAngleParameter(const std::string &name)
    : Parameter(name), m_var(name.c_str()) {
    m_sigma = std::numeric_limits<double>::infinity();
  }
  UniformAngleParameter(const std::string &name, parameter<double> pvar)
    : Parameter(name), m_var(pvar) {
    m_sigma = std::numeric_limits<double>::infinity();
  }

  void set(double value);
  double value() { return m_var; }
  double central() { return m_central; }
  void setCentral(double value) { m_central = value; }
  double sigma() { return m_sigma; }
  void setSigma(double sigma) { m_sigma = sigma; }
  double cast(const std::string &v) const;

  virtual double relativeValue(double diff) {
    return diff*std::numeric_limits<double>::infinity();
  }

  const variable<double> &getVariable() const { return m_var; }
protected:
  parameter<double> m_var;
  double m_central;
  double m_sigma;

  ClassDef(UniformAngleParameter, 0);
};

#endif // UNCERTAINPARAMETER_H
