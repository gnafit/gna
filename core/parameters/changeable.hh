#pragma once

#include <typeinfo>
#include <functional>
#include <deque>
#include <set>
#include <cstring>

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

#include "parameters_debug.hh"

class changeable {
public:
  void subscribe(changeable d) {
    inconstant_header *hdr = m_data.hdr;
    if (hdr->observers.has(d)) {
      DPRINTF("%p/%s is already observer", d.m_data.raw, d.name());
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
    //if(m_data.hdr->tainted){
      //DPRINTF("already tainted (do not propagate)");
      //return;
    //}
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
  changeable() = default;
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
    DPRINTF("subscribing to %i deps", int(deps.size()));
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

