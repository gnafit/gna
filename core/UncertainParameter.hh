#ifndef UNCERTAINPARAMETER_H
#define UNCERTAINPARAMETER_H

#include <string>
#include <limits>
#include <utility>
#include <map>
#include <unordered_map>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>

#include "Parameters.hh"
#include "GNAObject.hh"

#define COVARIANCE_DEBUG

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
class Uncertain: public GNASingleObject,
                 public Transformation<Uncertain<T>>
{
public:
  Uncertain(const std::string &name)
    : m_varhandle(variable_(&m_var, name)), m_name(name)
  { }
  Uncertain(const std::string &name, variable<void> var)
    : Uncertain(name)
  { m_varhandle.bind(variable<T>(var)); }

  virtual ~Uncertain() { }

  const std::string &name() const { return m_name; }
  virtual T value() const { return m_var.value(); }
  virtual T central() const { return m_central; }
  virtual void setCentral(T value) { m_central = value; }
  virtual T sigma() const { return m_sigma; }
  virtual void setSigma(T sigma) { m_sigma = sigma; }
  virtual const variable<T>& getVariable() const { return m_var; }
protected:
  variable<T> m_var;
  ParametrizedTypes::VariableHandle<T> m_varhandle;
  std::string m_name;
  T m_central;
  T m_sigma;
};

template <>
inline Uncertain<double>::Uncertain(const std::string &name)
  : m_varhandle(variable_(&m_var, name)), m_name(name)
{
  transformation_(this, "value")
    .output(name)
    .types([](Atypes, Rtypes rets) {
        rets[0] = DataType().points().shape(1);
      })
    .func([](Uncertain<double> *obj, Args, Rets rets) {
        rets[0].arr(0) = obj->m_var.value();
      })
    ;
}


template <typename T>
struct ParameterComparator {
    bool operator()(const T& lhs, const T& rhs) const {
        return lhs.value() < rhs.value();
    };
};

template <typename T>
class Parameter: public Uncertain<T> {
public:
  Parameter(const std::string &name)
    : Uncertain<T>(name)
    { m_par = this->m_varhandle.claim(); }

  static_assert(std::is_floating_point<T>::value, "Trying to use not floating point values in Parameter template");

  friend bool operator < (const Parameter<T>& lhs, const Parameter<T>& rhs)
  { return (lhs.value() < rhs.value()) || (lhs.name() < rhs.name());};

  virtual void set(T value)
    { m_par = value; }
  virtual T relativeValue(T diff)
    { return this->value() + diff*this->m_sigma; }
  virtual void relativeShift(T diff)
    { set(relativeValue(diff)); }

  virtual T cast(const std::string &v) const
    { return boost::lexical_cast<T>(v); }
  virtual T cast(const T &v) const
    { return v; }

  virtual void addLimits(T min, T max)
    { m_limits.push_back(std::make_pair(min, max)); }
  virtual const std::vector<std::pair<T, T>> &limits() const
    { return m_limits; }

  virtual void reset() { set(this->central()); }
  bool influences(SingleOutput &out) const {
    return out.single().depends(this->getVariable());
  }

  virtual bool isFixed() const { return this->m_fixed; }
  virtual void setFixed() { this->m_fixed = true; }

  virtual bool isCovariated(const Parameter<T>& other) const {
      auto it = m_covariances.find(other);
      if (it == m_covariances.end()) { 
          return false;
      } else {
          return true;
      }
  }

  virtual void setCovariance(Parameter<T>& other, T cov) {
    if ( &other == this) {
        this->setSigma(std::sqrt(cov));
    }
    m_covariances[other] = cov;
    other.updateCovariance(*this, cov);
  }

  virtual void updateCovariance(const Parameter<T>& other, T cov) {
    m_covariances[other] = cov;
  }

  virtual T getCovariance(const Parameter<T>& other) {
      auto search = m_covariances.find(other);
      if (search != m_covariances.end()) {
          return search->second;
      } else {
#ifdef COVARIANCE_DEBUG
          auto msg = boost::format("Parameters %1% and %2% are not covariated"); 
          std::cout << msg % this->name() % other.name() << std::endl;
#endif
          return static_cast<T>(0.);
      }
  }

  virtual const parameter<T>& getParameter() { return m_par; }

protected:
  std::vector<std::pair<T, T>> m_limits;
  using CovStorage = std::map<Parameter<T>, T>;
  CovStorage m_covariances;
  parameter<T> m_par;
  bool m_fixed = false;
};


template <typename T>
class GaussianParameter: public Parameter<T> {
public:
  GaussianParameter(const std::string &name)
    : Parameter<T>(name) { }
};

template <typename T>
class UniformAngleParameter: public Parameter<T> {
public:
  UniformAngleParameter(const std::string &name)
    : Parameter<T>(name)
    { this->m_sigma = std::numeric_limits<T>::infinity(); }
  void set(T value) override;
  T cast(const std::string &v) const override;
  T cast(const T &v) const override { return Parameter<T>::cast(v); }
};

template <typename T>
inline void UniformAngleParameter<T>::set(T value) {
  const double pi = boost::math::constants::pi<T>();

  T v = value - static_cast<int>(value/pi/2)*2*pi;
  if (v >= pi) {
    v -= 2*pi;
  }
  else if (v<-pi){
    v += 2*pi;
  }
  Parameter<T>::set(v);
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
