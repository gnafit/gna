#pragma once

#include "variable.hh"

template <typename ValueType>
class parameter;

template <>
class parameter<void>: public variable<void> {
public:
  template <typename T>
  parameter(const parameter<T> &other)
    : variable<void>(other) { }
  static parameter<void> null() {
    return parameter<void>();
  }
protected:
  parameter()
    : variable<void>() { }
};

template <typename ValueType>
class parameter: public variable<ValueType> {
  using base_type = variable<ValueType>;
public:
  parameter(const char *name="") : base_type() {
    base_type::m_data.raw = new inconstant_data<ValueType>(name);
  }
  parameter(const parameter<ValueType> &other)
    : base_type(other) { }
  explicit parameter(const parameter<void> &other)
    : variable<ValueType>(other) { }
  static parameter<ValueType> null() {
    return parameter<ValueType>(base_type::null());
  }
  parameter(std::initializer_list<const char*> name)
    : base_type(*name.begin()) { }
  parameter<ValueType>& operator=(ValueType v) {
    set(v);
    return *this;
  }
  void set(ValueType v) {
    DPRINTF("setting to %e", v);
    auto &d = base_type::data();
    if (d.value != v) {
      d.value = v;
      base_type::notify();
    }
    d.tainted = false;
  }
protected:
  parameter(const base_type &other)
    : base_type(other) { }
};
