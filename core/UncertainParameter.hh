#ifndef UNCERTAINPARAMETER_H
#define UNCERTAINPARAMETER_H

#include <string>
#include <limits>
#include <utility>

#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>

#include "Parameters.hh"

template <typename T>
class ParameterWrapper {
public:
  ParameterWrapper(const std::string &name)
    : m_var(name.c_str()) { }
  ParameterWrapper(parameter<void> &pvar)
    : m_var(pvar) { }

  T value() const { return m_var; }
  void set(T value) { m_var = value; }

  const variable<T> &getVariable() const { return m_var; }
protected:
  parameter<T> m_var;
};

template <typename T>
class Uncertain {
public:
  Uncertain(const std::string &name)
    : m_name(name) { }
  virtual ~Uncertain() { }

  const std::string &name() const { return m_name; }
  virtual T value() = 0;
  virtual T central() = 0;
  virtual void setCentral(T value) = 0;
  virtual T sigma() = 0;
  virtual void setSigma(T sigma) = 0;
protected:
  std::string m_name;
};

template <typename T>
class Parameter: public Uncertain<T> {
public:
  Parameter(const std::string &name)
    : Uncertain<T>(name) { }

  virtual void set(T value) = 0;
  virtual T relativeValue(T diff) = 0;
  virtual void relativeShift(T diff) { set(relativeValue(diff)); }

  virtual T cast(const std::string &v) const;
  virtual T cast(const T &v) const { return v; }

  virtual void addLimits(T min, T max)
    { m_limits.push_back(std::make_pair(min, max)); }
  virtual const std::vector<std::pair<T, T>> &limits() const
    { return m_limits; }

  using Uncertain<T>::central;
  virtual void reset() { set(central()); }
protected:
  std::vector<std::pair<T, T>> m_limits;
};

template <typename T>
T Parameter<T>::cast(const std::string &v) const {
  return boost::lexical_cast<T>(v);
}

template <typename T>
class GaussianParameter: public Parameter<T> {
public:
  GaussianParameter(const std::string &name)
    : Parameter<T>(name), m_var(name.c_str()) { }
  GaussianParameter(const std::string &name, parameter<void> pvar)
    : Parameter<T>(name), m_var(pvar) { }

  T value() { return m_var;}
  T central() { return m_central; }
  void setCentral(T value) { m_central = value; }
  T sigma() { return m_sigma; }
  void setSigma(T sigma) { m_sigma = sigma; }
  void set(T value) { m_var = value;}
  T relativeValue(T diff) { return value() + diff*m_sigma; }

  const variable<T> &getVariable() const { return m_var; }
protected:
  parameter<T> m_var;
  T m_central;
  T m_sigma;
};

template <typename T>
class GaussianValue: public Uncertain<T> {
public:
  GaussianValue(const std::string &name, variable<void> var)
    : Uncertain<T>(name), m_var(var) { }

  T value() { return m_var; }
  T central() { return m_central; }
  void setCentral(T value) { m_central = value; }
  T sigma() { return m_sigma; }
  void setSigma(T sigma) { m_sigma = sigma; }

  const variable<T> &getVariable() const { return m_var; }
protected:
  variable<T> m_var;
  T m_central;
  T m_sigma;
};

template <typename T>
class UniformAngleParameter: public Parameter<T> {
public:
  UniformAngleParameter(const std::string &name)
    : Parameter<T>(name), m_var(name.c_str()) {
    m_sigma = std::numeric_limits<T>::infinity();
  }
  UniformAngleParameter(const std::string &name, parameter<void> pvar)
    : Parameter<T>(name), m_var(pvar) {
    m_sigma = std::numeric_limits<T>::infinity();
  }

  void set(T value);
  T value() { return m_var; }
  T central() { return m_central; }
  void setCentral(T value) { m_central = value; }
  T sigma() { return m_sigma; }
  void setSigma(T sigma) { m_sigma = sigma; }
  T cast(const std::string &v) const;
  T cast(const T &v) const { return v; }

  virtual T relativeValue(T diff) {
    return diff*std::numeric_limits<T>::infinity();
  }

  const variable<T> &getVariable() const { return m_var; }
protected:
  parameter<T> m_var;
  T m_central;
  T m_sigma;
};

template <typename T>
inline void UniformAngleParameter<T>::set(T value) {
  const double pi = boost::math::constants::pi<T>();

  T v = value - static_cast<int>(value/pi)*2*pi;
  if (v > pi) {
    v -= 2*pi;
  }
  m_var = v;
}

template <typename T>
inline T UniformAngleParameter<T>::cast(const std::string &v) const {
  try {
    return boost::lexical_cast<T>(v);
  } catch (const boost::bad_lexical_cast&) {
  }
  const double pi = boost::math::constants::pi<T>();
  size_t pipos = v.find("pi");
  if (pipos == std::string::npos) {
    throw std::runtime_error("invalid uniform angle value");
  }
  T a = boost::lexical_cast<T>(v.substr(0, pipos));
  T b = boost::lexical_cast<T>(v.substr(pipos+2));
  return a*pi/b;
}

#endif // UNCERTAINPARAMETER_H
