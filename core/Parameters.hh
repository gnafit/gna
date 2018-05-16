#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <functional>
#include <typeinfo>
#include <stdexcept>
#include <set>
#include <deque>
#include <cstring>

#include "demangle.hpp"

class changeable;

struct references {
  bool has(changeable obj) const;
  void add(changeable obj);
  void resize(size_t newsize);
  void replace(changeable o, changeable n);
  void remove(changeable obj);

  changeable *objs = nullptr;
  size_t cnt = 0;
  size_t bufsize = 0;
};

struct inconstant_header {
  const char *name = nullptr;
  references observers;
  references emitters;
  bool tainted = true;
  std::function<void()> changed;
  const std::type_info *type = nullptr;
};

template <typename ValueType>
struct inconstant_data: public inconstant_header {
  inconstant_data() {
    type = &typeid(ValueType);
  }
  ValueType value;
  std::function<ValueType()> func;
};

// #define DEBUG_PARAMETERS

#ifdef DEBUG_PARAMETERS
#define DPRINTF(...) do {                                               \
    const changeable &c = *static_cast<const changeable*>(this);        \
    fprintf(stderr, "[%p/%s] ", c.rawdata(), c.name());                 \
    fprintf(stderr, __VA_ARGS__);                                       \
    fprintf(stderr, "\n");                                              \
  } while (0)
#else
#define DPRINTF(...)
#endif

class changeable {
public:
  void subscribe(changeable d) {
    inconstant_header *hdr = m_data.hdr;
    if (hdr->observers.has(d)) {
      return;
    }
    DPRINTF("%p/%s becomes observer", d.m_data.raw, d.name());
    hdr->observers.add(d);
    d.m_data.hdr->emitters.add(*this);
  }
  void unsubscribe(changeable d) {
    m_data.hdr->observers.remove(d);
    d.m_data.hdr->emitters.remove(*this);
  }
  const void *rawdata() const {
    return m_data.raw;
  }
  const void *rawptr() const {
    return rawdata();
  }
  const char *name() const {
    return m_data.hdr->name;
  }
  bool operator==(changeable const &other) {
    return m_data.raw == other.m_data.raw;
  }
  bool is(changeable const &other) {
    return *this == other;
  }
  bool isnull() const {
    return !m_data.raw;
  }
  void taint() const {
    DPRINTF("got tainted");
    notify();
    m_data.hdr->tainted = true;
    if (m_data.hdr->changed) {
      DPRINTF("calling changed callback");
      m_data.hdr->changed();
    }
  }
  bool depends(changeable other) const {
    std::deque<changeable> queue;
    std::set<const void*> visited;
    queue.push_back(*this);
    while (!queue.empty()) {
      changeable x = queue.front();
      queue.pop_front();
      references &ems = x.m_data.hdr->emitters;
      if (ems.has(other)) {
        return true;
      }
      if (visited.insert(x.rawptr()).second) {
        for (size_t i = 0; i < ems.cnt; ++i) {
          queue.push_back(ems.objs[i]);
        }
      }
    }
    return false;
  }
  void replace(changeable other) {
    if (this->is(other)) {
      return;
    }
    references &obs = m_data.hdr->observers;
    for (size_t i = 0; i < obs.cnt; ++i) {
      if (!obs.objs[i].isnull()) {
        other.m_data.hdr->observers.add(obs.objs[i]);
        obs.objs[i].m_data.hdr->emitters.replace(*this, other);
      }
    }
    references &ems = m_data.hdr->emitters;
    for (size_t i = 0; i < ems.cnt; ++i) {
      if (!ems.objs[i].isnull()) {
        other.m_data.hdr->emitters.add(ems.objs[i]);
        ems.objs[i].m_data.hdr->observers.replace(*this, other);
      }
    }
    assign(other);
  }
  void assign(changeable other) {
    m_data.raw = other.m_data.raw;
  }
protected:
  changeable() { }
  void notify() const {
    size_t nsubs = m_data.hdr->observers.cnt;
    changeable *observers = m_data.hdr->observers.objs;
    if (nsubs == 0) {
      return;
    }
    for (size_t i = 0; i < nsubs; ++i) {
      if (observers[i].isnull()) {
        continue;
      }
      observers[i].taint();
    }
  }
  template <typename T>
  void initdeps(T deps) {
    for (auto dep: deps) {
      dep.subscribe(*this);
    }
  }
  void init() {
    m_data.raw = new inconstant_header;
    DPRINTF("constructed header");
  }
  template <typename T>
  void init(T deps) {
    init();
    for (auto dep: deps) {
      dep.subscribe(*this);
    }
  }
  union {
    void *raw;
    inconstant_header *hdr;
  } m_data;
};

inline bool references::has(changeable obj) const {
  for (size_t i = 0; i < cnt; ++i) {
    if (objs[i] == obj) {
      return true;
    }
  }
  return false;
}

inline void references::add(changeable obj) {
  if (bufsize <= cnt) {
    size_t newsize = cnt + 8 + bufsize/2;
    resize(newsize);
  }
  objs[cnt++] = obj;
}

inline void references::resize(size_t newsize) {
  objs = static_cast<changeable*>(realloc(objs, newsize*sizeof(changeable)));
  if (!objs) {
    throw std::runtime_error("out of memory");
  }
  bufsize = newsize;
}

inline void references::replace(changeable o, changeable n) {
  for (size_t i = 0; i < cnt; ++i) {
    if (objs[i] == o) {
      objs[i] = n;
    }
  }
}

inline void references::remove(changeable d) {
  if (cnt == 0) {
    return;
  }
  size_t i = 0;
  for (i = 0; i < cnt; ++i) {
    if (objs[i] == d) {
      break;
    }
  }
  if (cnt > i+1) {
    std::memmove(objs+i, objs+i+1, cnt-(i+1));
  }
  cnt--;
}

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
  typedef variable<ValueType> base_type;
public:
  parameter() : base_type() {
    base_type::m_data.raw = new inconstant_data<ValueType>;
  }
  parameter(const parameter<ValueType> &other)
    : base_type(other) { }
  explicit parameter(const parameter<void> &other)
    : variable<ValueType>(other) { }
  static parameter<ValueType> null() {
    return parameter<ValueType>(base_type::null());
  }
  parameter(const char *name)
    : parameter() {
    base_type::data().name = name;
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
  evaluable(const char *name = 0) {
    base_type::m_data.raw = new inconstant_data<ValueType>;
    base_type::m_data.hdr->name = name;
    DPRINTF("constructed evaluable");
  }
  evaluable(const base_type &other)
    : base_type(other) { }
  static evaluable<ValueType> null() {
    return evaluable<ValueType>(base_type::null());
  }
};

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
  typedef evaluable<ValueType> base_type;
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
            const char *name = nullptr)
    : base_type(name) { base_type::init(f, deps); }
  template <typename T>
  dependant(std::function<ValueType()> f,
            std::vector<T> deps,
            const char *name = nullptr)
    : base_type(name) { base_type::init(f, deps); }
protected:
  dependant(const base_type &other)
    : base_type(other) { }
};

class taintflag: public changeable {
public:
  taintflag() { init(); }
  taintflag(std::initializer_list<changeable> deps) { init(deps); }
  taintflag(std::vector<changeable> deps) { init(deps); }
  operator bool() const {
    return m_data.hdr->tainted;
  }
  taintflag &operator=(bool value) {
    if (!value) {
     m_data.hdr->tainted = false;
    }
    return *this;
  }
};

class callback: public changeable {
public:
  callback() { init(); }
  callback(std::function<void()> f, std::initializer_list<changeable> deps) {
    init(deps);
    m_data.hdr->changed = f;
  }
  template <typename T>
  callback(std::function<void()> f, std::vector<T> deps) {
    init(deps);
    m_data.hdr->changed = f;
  }
};

#endif // PARAMETERS_H
