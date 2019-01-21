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
  const std::type_info *type() const {
    return m_data.hdr->type;
  }
  std::string typeName() const {
    return boost::core::demangle(type()->name());
  }
  template <typename T>
  bool istype() {
    return (type()->hash_code() == typeid(T).hash_code());
  }
  bool sametype(const variable<void> &other) {
    return (type()->hash_code() == other.type()->hash_code());
  }
};

template <typename ValueType>
class variable: public variable<void> {
public:
  operator const ValueType&() const {
    return value();
  }
  variable() {
    m_data.raw = nullptr;
  }
  variable(const variable<ValueType> &other)
    : variable<void>(other) { }
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
    return data().value;
  }
protected:
  inline inconstant_data<ValueType> &data() const {
    return *static_cast<inconstant_data<ValueType>*>(m_data.raw);
  }
  void update() const {
    auto &d = data();
    if (!d.tainted) {
      return;
    }
    if (!d.func) {
      return;
    }
    d.value = d.func();
    d.tainted = false;
  }
};
