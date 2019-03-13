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
  using base_type = variable<ValueType>;
protected:
  template <typename T>
  void init(std::function<ValueType()> f, T deps) {
    auto &d = base_type::data();
    if(d.value.size()!=1u){
      throw std::runtime_error("unable to set scalar function for vector data");
    }
    d.func = f;
    this->initdeps(deps);
  }
  template <typename T>
  void init(std::function<void(arrayview<ValueType>&)> vf, T deps) {
    auto &d = base_type::data();
    d.vfunc = vf;
    this->initdeps(deps);
  }
  evaluable(const char *name="", size_t size=1u) {
    base_type::alloc(new inconstant_data<ValueType>(size, name));
    DPRINTF("constructed evaluable");
  }
  evaluable(const base_type &other)
    : base_type(other) { }
  static evaluable<ValueType> null() {
    return evaluable<ValueType>(base_type::null());
  }
};

