#pragma once

#include "parameter.hh"

template <typename ValueType>
class evaluable;

template <>
class evaluable<void>: public variable<void> {
public:
  template <typename T>
  evaluable(const evaluable<T> &other)
    : variable<void>(other) { }
  static evaluable<void> null() {
    return evaluable<void>();
  }
protected:
  evaluable() : variable<void>() { }
};

template <typename ValueType>
class evaluable: public variable<ValueType> {
  typedef variable<ValueType> base_type;
protected:
  template <typename T>
  void init(std::function<ValueType()> f, T deps) {
    auto &d = base_type::data();
    d.func = f;
    this->initdeps(deps);
  }
  evaluable(const char *name = "") {
    base_type::m_data.raw = new inconstant_data<ValueType>(name);
    DPRINTF("constructed evaluable");
  }
  evaluable(const base_type &other)
    : base_type(other) { }
  static evaluable<ValueType> null() {
    return evaluable<ValueType>(base_type::null());
  }
};

