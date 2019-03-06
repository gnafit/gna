#pragma once

#include <vector>
#include "evaluable.hh"

template <typename ValueType>
class dependant;

template <>
class dependant<void>: public evaluable<void> {
public:
  template <typename T>
  dependant(const dependant<T> &other)
    : evaluable<void>(other) { }
  static dependant<void> null() {
    return dependant<void>();
  }
protected:
  dependant() : evaluable<void>() { }
};

template <typename ValueType>
class dependant: public evaluable<ValueType> {
  using base_type = evaluable<ValueType>;
public:
  dependant(const variable<ValueType> &other)
    : base_type(other) { }
  static dependant<ValueType> null() {
    return dependant<ValueType>(base_type::null());
  }
  dependant()
    : base_type(base_type::null()) { }
  dependant(std::function<ValueType()> f,
            std::initializer_list<changeable> deps,
            const char *name = "", size_t size=1u)
    : base_type(name, size)
    {
      base_type::init(f, deps);
      DPRINTF("constructed dependant");
    }
  template <typename T>
  dependant(std::function<ValueType()> f,
            std::vector<T> deps,
            const char *name = "", size_t size=1u)
    : base_type(name, size)
    {
      DPRINTF("construct dependant: %i deps", int(deps.size()));
      base_type::init(f, deps);
      DPRINTF("constructed dependant");
    }
  dependant(std::function<void(std::vector<ValueType>&)> vf,
            std::initializer_list<changeable> deps,
            const char *name = "", size_t size=1u)
    : base_type(name, size)
    {
      base_type::init(vf, deps);
      DPRINTF("constructed dependant");
    }
  template <typename T>
  dependant(std::function<void(std::vector<ValueType>&)> vf,
            std::vector<T> deps,
            const char *name = "", size_t size=1u)
    : base_type(name, size)
    {
      DPRINTF("construct dependant: %i deps", int(deps.size()));
      base_type::init(vf, deps);
      DPRINTF("constructed dependant");
    }
protected:
  dependant(const base_type &other)
    : base_type(other) { }
};

