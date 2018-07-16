#pragma once

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

/* #define COVARIANCE_DEBUG */


template <typename T>
class ParameterWrapper {
public:
  ParameterWrapper(const std::string &name)
    : m_var(name.c_str()) { }
  ParameterWrapper(parameter<void> &pvar)
    : m_var(pvar) { }

  T value() const noexcept { return m_var; }
  void set(T value) { m_var = value; }

  const variable<T> &getVariable() const noexcept { return m_var; }
protected:
  parameter<T> m_var;
};

template <typename T>
class Variable: public GNASingleObject,
                public TransformationBind<Variable<T>>
{
public:
  Variable(const std::string &name)
    : m_varhandle(variable_(&m_var, name)), m_name(name)
  { }
  Variable(const std::string &name, variable<void> var)
    : Variable(name)
  { m_varhandle.bind(variable<T>(var)); }

  virtual ~Variable() { }

  const std::string &name() const { return m_name; }
  virtual T value() { return m_var.value(); }
  virtual const variable<T> &getVariable() const noexcept { return m_var; }

  const std::string& label() const noexcept { return transformations[0].label(); }
  void setLabel(const std::string& label) { transformations[0].setLabel(label); }
protected:
  variable<T> m_var;
  ParametrizedTypes::VariableHandle<T> m_varhandle;
  std::string m_name;
};

template <>
inline Variable<double>::Variable(const std::string &name)
  : m_varhandle(variable_(&m_var, name)), m_name(name)
{
  transformation_("value")
    .output(name)
    .types([](TypesFunctionArgs fargs) {
        fargs.rets[0] = DataType().points().shape(1);
      })
    .func([](Variable<double> *obj, FunctionArgs fargs) {
        fargs.rets[0].arr(0) = obj->m_var.value();
      })
    .finalize();
}


/* template <typename T>
 * struct ParameterComparator {
 *     bool operator()(const T& lhs, const T& rhs) const noexcept {
 *         return lhs.value() < rhs.value();
 *     };
 * }; */

template <typename T>
class Parameter: public Variable<T> {
public:
  Parameter(const std::string &name)
    : Variable<T>(name)
    { m_par = this->m_varhandle.claim(); }

  static_assert(std::is_floating_point<T>::value, "Trying to use not floating point values in Parameter template");

  friend bool operator < (const Parameter<T>& lhs, const Parameter<T>& rhs) noexcept
  { return (lhs.value() < rhs.value()) || (lhs.name() < rhs.name());};

  virtual void set(T value)
    { m_par = value; }

  virtual T cast(const std::string& v) const
    { return boost::lexical_cast<T>(v); }

  virtual T cast(const T& v) const
    { return v; }

  virtual T central() { return m_central; }
  virtual void setCentral(T value) { m_central = value; }
  virtual void reset() { set(this->central()); }

  virtual T step() { return m_step; }
  virtual void setStep(T step) { m_step = step; }

  virtual T relativeValue(T diff)
    { return this->value() + diff*this->m_step; }
  virtual void setRelativeValue(T diff)
    { set(relativeValue(diff)); }

  virtual void addLimits(T min, T max)
    { m_limits.push_back(std::make_pair(min, max)); }

  virtual const std::vector<std::pair<T, T>>& limits() const
    { return m_limits; }


  virtual bool influences(SingleOutput &out) const noexcept {
    return out.single().depends(this->getVariable());
  }

  virtual bool isFixed() const noexcept { return this->m_fixed; }
  virtual void setFixed() noexcept { this->m_fixed = true; }


  virtual const parameter<T>& getParameter() const noexcept { return m_par; }

protected:
  T m_central;
  T m_step;
  std::vector<std::pair<T, T>> m_limits;
  /* using CovStorage = std::map<const Parameter<T>*, T>;
   * CovStorage m_covariances; */
  parameter<T> m_par;
  bool m_fixed = false;

};


template <typename T>
class GaussianParameter: public Parameter<T> {
public:
  GaussianParameter(const std::string &name)
    : Parameter<T>(name) { }

  virtual T sigma() const noexcept { return m_sigma; }
  virtual void setSigma(T sigma) { this->m_sigma=sigma; this->setStep(sigma*0.1); }

  virtual bool isCovariated(const GaussianParameter<T>& other) const noexcept {
      auto it = this->m_covariances.find(&other);
      if (it == this->m_covariances.end() and (&other != this)) {
          return false;
      } else {
          return true;
      }
  }


  virtual void setCovariance(GaussianParameter<T>& other, T cov) {
#ifdef COVARIANCE_DEBUG
        auto msg = boost::format("Covariance of parameters %1% and %2% is set to %3%");
        std::cout << msg % this->name() % other.name() % cov << std::endl;
#endif
    if (&other != this) {
        this->m_covariances[&other] = cov;
        other.updateCovariance(*this, cov);
    } else {
        this->setSigma(std::sqrt(cov));
    }
  }

  virtual void updateCovariance(GaussianParameter<T>& other, T cov) {
#ifdef COVARIANCE_DEBUG
    auto msg = boost::format("Covariance of parameters %1% and %2% is updated "
                             "to %3% after setting in %1%");
    std::cout << msg % this->name() % other.name() % cov << std::endl;
#endif
    this->m_covariances[&other] = cov;
  }

  virtual T getCovariance(const GaussianParameter<T>& other) const noexcept {
      if (this == &other) {return this->sigma()*this->sigma();}
      auto search = m_covariances.find(&other);
      if (search != m_covariances.end()) {
          return search->second;
      } else  {
#ifdef COVARIANCE_DEBUG
          auto msg = boost::format("Parameters %1% and %2% are not covariated");
          std::cout << msg % this->name() % other.name() << std::endl;
#endif
          return static_cast<T>(0.);
      }
  }

  virtual T normalValue(T reldiff)
    { return this->central() + reldiff*this->m_sigma; }

  virtual void setNormalValue(T reldiff)
    { this->set(this->normalValue(reldiff)); }
protected:
  T m_sigma;
  using CovStorage = std::map<const GaussianParameter<T>*, T>;
  CovStorage m_covariances;
};

template <typename T>
class UniformAngleParameter: public Parameter<T> {
public:
  UniformAngleParameter(const std::string &name)
    : Parameter<T>(name)
    { this->setStep(0.017453292519943295); /*1 degree*/ }

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
