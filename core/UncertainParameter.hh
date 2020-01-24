#pragma once

#include <string>
#include <limits>
#include <utility>
#include <map>
#include <stack>
#include <cmath>

#include <boost/lexical_cast.hpp>
#include "fmt/format.h"
#include <boost/math/constants/constants.hpp>

#include "GNAObject.hh"
#include "parameters/parameter.hh"


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
class Variable: public GNASingleObjectT<T,T>,
                public TransformationBind<Variable<T>,T,T>
{
public:
  Variable(const std::string &name, bool allocate=false)
    : m_varhandle(variable_(&m_var, name, static_cast<size_t>(allocate))), m_name(name)
  {}

  Variable(const std::string &name, variable<void> var)
    : Variable(name)
  {
    m_varhandle.bind(variable<T>(var));
    this->transformations.front().updateTypes();
  }

  ~Variable() override = default;

  virtual T value() const noexcept { return m_var.value(); }
  const variable<T> &getVariable() const noexcept { return m_var; }

  std::string name() const noexcept { return m_name; }

  std::string inNamespace() const noexcept { return m_namespace; }
  void setNamespace(const std::string& ns_name) { m_namespace = ns_name; };
  std::string qualifiedName() const { return m_namespace + "." + m_name; };

  std::string label() const noexcept { return m_label; }
  void setLabel(const std::string& label) { m_label=label; }
  void setLabel(std::string&& label) { m_label = label; }

  size_t hash() const {return reinterpret_cast<size_t>(this);}

protected:
  variable<T> m_var{};
  ParametrizedTypes::VariableHandle<T> m_varhandle;
  std::string m_name;
  std::string m_label;
  std::string m_namespace;
};

template <>
inline Variable<double>::Variable(const std::string &name, bool allocate)
  : m_varhandle(variable_(&m_var, name, static_cast<size_t>(allocate))), m_name(name)
{
  transformation_("value")
    .output(name)
    .types([](Variable<double> *obj, TypesFunctionArgs& fargs) {
        fargs.rets[0] = DataType().points().shape(obj->m_var.size());
      })
    .func([](Variable<double> *obj, FunctionArgs& fargs) {
        obj->m_var.values(fargs.rets[0].buffer);
      })
    .finalize();
}

#ifdef PROVIDE_SINGLE_PRECISION
template <>
inline Variable<float>::Variable(const std::string &name, bool allocate)
  : m_varhandle(variable_(&m_var, name, static_cast<size_t>(allocate))), m_name(name)
{
  transformation_("value")
    .output(name)
    .types([](Variable<float> *obj, TypesFunctionArgs& fargs) {
        fargs.rets[0] = DataType().points().shape(obj->m_var.size());
      })
    .func([](Variable<float> *obj, FunctionArgs& fargs) {
        obj->m_var.values(fargs.rets[0].buffer);
      })
    .finalize();
}
#endif

template <typename T>
class Parameter: public Variable<T> {
public:
  Parameter(const std::string &name)
    : Variable<T>(name, true/*allocate*/), m_par(this->m_varhandle.claim())
    {
      this->transformations.front().updateTypes();
    }

  virtual void set(T value)
    { m_par = value; }

  T push() {
    m_stack.push(this->value());
    return this->value();
  }

  T push(T value) {
    m_stack.push(this->value());
    this->set(value);
    return value;
  }

  T pop() {
    this->set(m_stack.top());
    m_stack.pop();
    return this->value();
  }

  virtual T cast(const std::string& v) const
    { return boost::lexical_cast<T>(v); }

  virtual T cast(const T& v) const noexcept
    { return v; }

  T central() const noexcept { return m_central; }
  void setCentral(T value) noexcept { m_central = value; }
  void reset() { set(this->central()); }

  T step() const noexcept { return m_step; }
  void setStep(T step) noexcept { m_step = step; }

  T relativeValue(T diff) const noexcept
    { return this->value() + diff*this->m_step; }
  void setRelativeValue(T diff)
    { set(relativeValue(diff)); }

  void addLimits(T min, T max)
    { m_limits.push_back(std::make_pair(min, max)); }

  const std::vector<std::pair<T, T>>& limits() const
    { return m_limits; }

  bool influences(SingleOutput &out) const {
    return out.single().depends(this->getVariable());
  }

  bool isFixed() const noexcept { return this->m_fixed; }
  void setFixed() noexcept { this->m_fixed = true; }

  bool isFree() const noexcept { return this->m_free; }
  void setFree(bool free=true) noexcept { this->m_free = free; }

  const parameter<T>& getParameter() const noexcept { return m_par; }

protected:
  T m_central;
  T m_step;
  std::vector<std::pair<T, T>> m_limits;
  parameter<T> m_par;
  bool m_fixed = false;
  bool m_free = false;

  std::stack<T> m_stack;
};

template <typename T>
class GaussianParameter: public Parameter<T> {
public:
  GaussianParameter(const std::string &name)
    : Parameter<T>(name) { }
  std::vector<GaussianParameter<T>*> m_cov_pars{};

  T sigma() const noexcept { return m_sigma; }
  void setSigma(T sigma) noexcept {
    this->m_sigma=sigma;
    if(std::isinf(sigma)){
      this->setFree();
    }else{
      this->setFree(false);
      this->setStep(sigma*0.1);
    }
  }

   void setRelSigma(T relsigma) {
     auto central = this->central();
     if(central==static_cast<T>(0)){
       throw std::runtime_error("May not set relative uncertainty to the parameter with central=0");
     }
     this->setSigma(central*relsigma);
   }

  bool isCorrelated(const GaussianParameter<T>& other) const noexcept {
    auto it = this->m_covariances.find(&other);
    if (it == this->m_covariances.end() and (&other != this)) {
      return false;
    } else {
      return true;
    }
  }

  bool isCorrelated() const noexcept {
    return !m_covariances.empty();
  }

  std::vector<GaussianParameter<T>*>  getAllCorrelatedWith() const {
    std::vector<GaussianParameter<T>*> tmp;
    for (const auto& item: this->m_covariances){
      tmp.push_back(const_cast<GaussianParameter<T>*>(item.first));
    }
    return tmp;
  }


  void setCovariance(GaussianParameter<T>& other, T cov) {
#ifdef COVARIANCE_DEBUG
    fmt::print("Covariance of parameters {0} and {1} is set to {2}",
               this->name(), other.name(), cov);
#endif
    if (&other != this) {
      this->m_covariances[&other] = cov;
      other.updateCovariance(*this, cov);
    } else {
      this->setSigma(std::sqrt(cov));
    }
  }


  void updateCovariance(GaussianParameter<T>& other, T cov) {
#ifdef COVARIANCE_DEBUG
    fmt::print("Covariance of parameters {0} and {1} is updated "
               "to {2} after setting in {0}", this->name(), other.name(), cov);
#endif
    this->m_covariances[&other] = cov;
  }

  T getCovariance(const GaussianParameter<T>& other) const noexcept {
    if (this == &other) {return this->sigma()*this->sigma();}
    auto search = m_covariances.find(&other);
    if (search != m_covariances.end()) {
      return search->second;
    } else  {
#ifdef COVARIANCE_DEBUG
      fmt::print("Parameters {0} and {1} are not covariated", this->name(), other.name());
#endif
      return static_cast<T>(0.);
    }
  }

  T getCorrelation(const GaussianParameter<T>& other) const noexcept {
    if (this == &other) {return static_cast<T>(1);}
    auto search = m_covariances.find(&other);
    if (search != m_covariances.end()) {
      return search->second / (this->sigma() * other.sigma());
    } else  {
#ifdef COVARIANCE_DEBUG
      fmt::print("Parameters {0} and {1} are not covariated", this->name(), other.name());
#endif
      return static_cast<T>(0.);
    }
  }

  T normalValue(T reldiff) const noexcept
  { return this->central() + reldiff*this->m_sigma; }

  void setNormalValue(T reldiff)
  { this->set(this->normalValue(reldiff)); }
protected:
  using CovStorage = std::map<const GaussianParameter<T>*, T>;
  T m_sigma;
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
  T cast(const T &v) const noexcept override { return Parameter<T>::cast(v); }
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
