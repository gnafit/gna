#pragma once

#include <string>
#include "demangle.hpp"
#include "changeable.hh"

template <typename ValueType>
class variable;

template <>
class variable<void>: public changeable {
public:
  static variable<void> null() {
    return variable<void>();
  }
  const std::type_info *type() const noexcept {
    return m_hdr->type;
  }
  std::string typeName() const {
    return boost::core::demangle(type()->name());
  }
  template <typename T>
  bool istype() const {
    return (type()->hash_code() == typeid(T).hash_code());
  }
  bool sametype(const variable<void> &other) const {
    return (type()->hash_code() == other.type()->hash_code()) && ( size()==other.size() );
  }
  size_t size() const noexcept {
    return m_hdr->size;
  }
};

template <typename ValueType>
class variable: public variable<void> {
public:
  operator const ValueType&() const {
    return value();
  }
  variable(){};
  variable(const variable<ValueType> &other)
    : variable<void>(other) {
    }
  explicit variable(const variable<void> &other)
    : variable<void>(other) {
    if (!this->sametype(other)) {
      throw std::runtime_error("bad variable conversion");
    }
  }
  static variable<ValueType> null() {
    return variable<ValueType>();
  }
  const ValueType &value() const {
    update();
    return data().value[0];
  }
  const std::complex<ValueType> &complex() const {
    update();
    return data().value.complex();
  }
  const ValueType &value(size_t i) const {
    update();
    return data().value[i];
  }
  const arrayview<ValueType> &values() const {
    update();
    return data().value;
  }
  void values(std::vector<ValueType>& dest) const {
    update();
    auto& value = data().value;
    dest.resize(value.size());
    std::copy(value.begin(), value.end(), dest.begin());
  }
  ValueType* values(ValueType* dest) const {
    update();
    auto& val=data().value;
    std::copy(val.begin(), val.end(), dest);
    return std::next(dest, val.size());
  }
  bool hasFunc() const {
    auto& d = data();
    return d.func || d.vfunc;
  }
  void update() const {
    auto &d = data();
    if (!d.tainted) {
      return;
    }
    d.tainted = false;
    if (d.func) {
      d.value[0] = d.func();
    }
    else if (d.vfunc) {
      d.vfunc(d.value);
    }
  }
protected:
  inline inconstant_data<ValueType> &data() const {
    return *static_cast<inconstant_data<ValueType>*>(m_hdr.get());
  }
};

