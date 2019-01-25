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
  parameter(const char *name="", size_t size=1u) : base_type() {
    base_type::alloc(new inconstant_data<ValueType>(size, name));
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
    if (d.value[0] != v) {
      d.value[0] = v;
      base_type::notify();
    }
    d.tainted = false;
  }
  void set(size_t i, ValueType v) {
    DPRINTF("setting [%zu] to %e", i, v);
    auto &d = base_type::data();
    if (d.value[i] != v) {
      d.value[i] = v;
      base_type::notify();
    }
    d.tainted = false;
  }
  void set(const std::vector<ValueType>& other) {
    auto &d = base_type::data();
    if (d.value != other) {
      d.value = other;
      base_type::notify();
    }
    d.tainted = false;
  }
  void set(ValueType* other) {
    auto &d = base_type::data();
    auto& values=d.value;
    if( !std::equal(values.begin(), values.end(), other) ){
        base_type::notify();
        std::copy(other, std::next(other, values.size(), values.data()));
    }
    d.tainted = false;
  }
protected:
  parameter(const base_type &other)
    : base_type(other) { }
};
