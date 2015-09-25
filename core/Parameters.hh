#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <functional>

class changeable;
struct references {
  bool has(changeable obj) const;
  void add(changeable obj);
  void resize(size_t newsize);
  void replace(changeable o, changeable n);

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
};

template <typename ValueType>
struct inconstant_data: public inconstant_header {
  ValueType value;
  std::function<ValueType()> func;
};

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
  const void *rawdata() const {
    return m_data.raw;
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

template <typename ValueType>
class variable: public changeable {
public:
  operator const ValueType&() const {
    update();
    return data().value;
  }
  variable() {
    m_data.raw = nullptr;
  }
  variable(const variable<ValueType> &other) {
    m_data.raw = other.m_data.raw;
  }
  static variable<ValueType> null() {
    return variable<ValueType>();
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
class independant: public variable<ValueType> {
  typedef variable<ValueType> base_type;
public:
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
  independant() : base_type() {
    base_type::m_data.raw = new inconstant_data<ValueType>;
  }
  independant(const char *name) : independant() {
    base_type::data().name = name;
    DPRINTF("constructed independant");
  }
  independant(const base_type &other)
    : base_type(other) { }
  independant(const independant<ValueType> &other)
    : base_type(other) { }
  static independant<ValueType> null() {
    return independant<ValueType>(base_type::null());
  }
};

template <typename ValueType>
class parameter: public independant<ValueType> {
  typedef independant<ValueType> base_type;
public:
  parameter(const parameter<ValueType> &other)
    : base_type(other) { }
  static parameter<ValueType> null() {
    return parameter<ValueType>(base_type::null());
  }
  parameter(const char *name = nullptr)
    : base_type(name) { }
  parameter(std::initializer_list<const char*> name)
    : base_type(*name.begin()) { }
  parameter<ValueType>& operator=(ValueType v) {
    base_type::set(v);
    return *this;
  }
protected:
  parameter(const base_type &other)
    : base_type(other) { }
};

template <typename ValueType>
class freevar: public independant<ValueType> {
  typedef independant<ValueType> base_type;
public:
  freevar(const freevar<ValueType> &other)
    : base_type(other) { }
  static freevar<ValueType> null() {
    return freevar<ValueType>(base_type::null());
  }
  freevar(const char *name = nullptr)
    : base_type(name) { }
  freevar<ValueType>& operator=(ValueType v) {
    base_type::set(v);
    return *this;
  }
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

template <typename ValueType>
class functional: public evaluable<ValueType> {
  typedef evaluable<ValueType> base_type;
public:
  functional(const functional<ValueType> &other)
    : base_type(other) { }
  static functional<ValueType> null() {
    return functional<ValueType>(base_type::null());
  }
  functional()
    : base_type(base_type::null()) { }
  functional(std::function<ValueType()> f,
             std::initializer_list<changeable> deps,
             const char *name = nullptr)
    : base_type(name) { base_type::init(f, deps); }
  template <typename T>
  functional(std::function<ValueType()> f,
             std::vector<T> deps,
             const char *name = nullptr)
    : base_type(name) { base_type::init(f, deps); }
protected:
  functional(const base_type &other)
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
