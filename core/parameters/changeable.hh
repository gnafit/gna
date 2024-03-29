#pragma once

#include <typeinfo>
#include <functional>
#include <deque>
#include <initializer_list>
#include <set>
#include <cstring>
#include <string>

#include "arrayview.hh"

class changeable;

struct references {
  using Container = std::vector<changeable>;

  bool   has(changeable obj) const;
  size_t index(changeable obj) const;
  void   add(changeable obj);
  void   replace(changeable o, changeable n);
  void   remove(changeable obj);
  size_t size() { return objs.size(); }

  Container::iterator begin() { return objs.begin(); }
  Container::iterator end()   { return objs.end(); }

  Container objs;
};

enum class TaintStatus {
  Normal = 0,    ///<
  Frozen,        ///<
  FrozenTainted, ///<
  PassThrough    ///<
};

struct inconstant_header {
  inconstant_header(const char* aname="", const char* alabel="", bool autoname=false, size_t size=1u) : name(aname), label(alabel), size(size) {
    static size_t ih_counter=0;
    autoname = autoname || !name.size();
    if(autoname){
      name=name+"_"+std::to_string(ih_counter);
      ++ih_counter;
    }
  }
  virtual ~inconstant_header()=default;
  std::string name = "";
  std::string label = "";
  references observers;
  references emitters;
  bool tainted = true;
  TaintStatus status=TaintStatus::Normal;
  std::function<void()> on_taint;
  const std::type_info *type = nullptr;
  size_t size=1u;
};

template <typename ValueType>
struct inconstant_data: public inconstant_header {
  inconstant_data(size_t size=1u, const char* name="", const char* label="", bool autoname=false) : inconstant_header(name, label, autoname, size), value(size) {
    //printf("make %s of size %zu\n", this->name.c_str(), size);
    type = &typeid(ValueType);
  }
  arrayview<ValueType> value;
  std::function<ValueType()> func{nullptr};
  std::function<void(arrayview<ValueType>&)> vfunc{nullptr};
};

#include "parameters_debug.hh"

class changeable {
public:
  void subscribe(changeable d) {
    if (m_hdr->observers.has(d)) {
      DPRINTF("%p/%s is already observer", static_cast<void*>(d.m_hdr), d.name());
      return;
    }
    DPRINTF("%p/%s becomes observer", static_cast<void*>(d.m_hdr), d.name());
    m_hdr->observers.add(d);
    d.m_hdr->emitters.add(*this);
    if(m_hdr->tainted){
      DPRINTF("taint %p/%s", static_cast<void*>(d.m_hdr), d.name());
      d.taint();
    }
  }
  void subscirbeto(changeable d){
    d.subscribe(*this);
  }
  void unsubscribe(changeable d) {
    m_hdr->observers.remove(d);
    d.m_hdr->emitters.remove(*this);
  }
  void unsubscribefrom(changeable d) {
    d.unsubscribe(*this);
  }
  void *rawdata() const {
    return static_cast<void*>(m_hdr.get());
  }
  size_t hash() const {
    return reinterpret_cast<size_t>(rawdata());
  }
  const char *name() const {
    return m_hdr->name.c_str();
  }
  const char *label() const {
    return m_hdr->label.c_str();
  }
  bool operator==(changeable const &other) const {
    return m_hdr == other.m_hdr;
  }
  bool is(changeable const &other) {
    return *this == other;
  }
  bool isnull() const {
    return !m_hdr;
  }
  bool tainted() const {
    return m_hdr->tainted;
  }
  bool frozen() const {
    return m_hdr->status==TaintStatus::Frozen || m_hdr->status==TaintStatus::FrozenTainted;
  }
  bool passthrough() const {
    return m_hdr->status==TaintStatus::PassThrough;
  }
  TaintStatus taintstatus() const {
    return m_hdr->status;
  }
  void taint() const {
    if(frozen()){
      m_hdr->status=TaintStatus::FrozenTainted;
      return;
    }
    else{
      if(m_hdr->tainted && m_hdr->status!=TaintStatus::PassThrough){
        DPRINTF("already tainted (do not propagate)");
        return;
      }
    }
    DPRINTF("got tainted");
    notify();
    m_hdr->tainted = true;
    if (m_hdr->on_taint) {
      DPRINTF("calling on_taint callback");
      m_hdr->on_taint();
    }
  }
  void freeze(){
    switch(m_hdr->status){
      case TaintStatus::Normal:
        if(!m_hdr->tainted){
          m_hdr->status=TaintStatus::Frozen;
        }else{
          throw std::runtime_error("can not freeze tainted changeable");
        }
        break;
      case TaintStatus::PassThrough:
        throw std::runtime_error("can not freeze PassThrough changeable");
        break;
      default:
        break;
    }
  }
  void unfreeze(){
    if(!frozen()) return;
    auto wastainted = m_hdr->status==TaintStatus::FrozenTainted;
    m_hdr->status=TaintStatus::Normal;
    if(wastainted){
      taint();
    }
  }
  bool depends(changeable other) const {
    if(frozen()){
      return false;
    }
    if(*this==other){
      return true;
    }
    std::deque<changeable> queue;
    std::set<const void*> visited;
    queue.push_back(*this);
    while (!queue.empty()) {
      changeable current = queue.front();
      queue.pop_front();
      if(current.frozen()){
        continue;
      }
      references &emitters = current.m_hdr->emitters;
      if (emitters.has(other)) {
        return true;
      }
      if (visited.insert(current.rawdata()).second) {
        queue.insert(queue.end(), emitters.begin(), emitters.end());
      }
    }
    return false;
  }
  size_t distance(changeable other, bool skip_passthrough=false, size_t max_depth=-1lu) const {
    if(*this==other){
      return 0u;
    }
    if(!max_depth){
      return -1lu;
    }
    std::deque<std::pair<changeable*,size_t>> queue;
    std::set<const void*> visited;
    queue.push_back({const_cast<changeable*>(this), 1lu});
    changeable* current;
    size_t depth;
    while (!queue.empty()) {
      std::tie(current, depth) = queue.front();
      queue.pop_front();
      references &emitters = current->m_hdr->emitters;
      if (emitters.has(other)) {
        return depth;
      }
      if (visited.insert(current->rawdata()).second) {
        for(auto& emitter: emitters){
          size_t newdepth = skip_passthrough ? depth+static_cast<size_t>(!emitter.passthrough()) : depth+1;
          if(newdepth>max_depth){
            continue;
          }
          queue.push_back({&emitter, newdepth});
        }
      }
    }
    return -1lu;
  }
  void replace(changeable other) {
    if (this->is(other)) {
      return;
    }
    references &obs = m_hdr->observers;
    for (auto& obj: obs) {
      if (!obj.isnull()) {
        other.m_hdr->observers.add(obj);
        obj.m_hdr->emitters.replace(*this, other);
      }
    }
    references &ems = m_hdr->emitters;
    for (auto& obj: ems) {
      if (!obj.isnull()) {
        other.m_hdr->emitters.add(obj);
        obj.m_hdr->observers.replace(*this, other);
      }
    }
    assign(other);
    if(m_hdr->tainted){
      notify();
    }
  }
  void assign(changeable other) {
    m_hdr = other.m_hdr;
  }
  void notify() const {
    for (auto& observer: m_hdr->observers) {
      if (observer.isnull()) {
        continue;
      }
      observer.taint();
    }
  }
protected:
  changeable() = default;
  template <typename T>
  void initdeps(T deps) {
    DPRINTF("subscribing to %i deps", int(deps.size()));
    for (auto dep: deps) {
      dep.subscribe(*this);
    }
  }
  void alloc(inconstant_header* hdr){
    if(m_hdr){
      throw std::runtime_error("double initialization of changeable");
    }
    m_hdr.reset(hdr);
  }
  void init(const char* name="", bool autoname=false, size_t size=0u) {
    alloc(new inconstant_header(name, "", autoname, size));
    DPRINTF("constructed header");
  }
  template <typename T>
  void init(T deps, const char* name="") {
    init(name);
    for (auto dep: deps) {
      dep.subscribe(*this);
    }
  }
  std::shared_ptr<inconstant_header> m_hdr;
};

inline bool references::has(changeable obj) const {
  for (auto& o: objs) {
    if (o == obj) {
      return true;
    }
  }
  return false;
}

inline size_t references::index(changeable obj) const {
  for (size_t i(0); i<objs.size(); ++i) {
    if (objs[i] == obj) {
      return i;
    }
  }
  return -1u;
}

inline void references::add(changeable obj) {
  if(has(obj)){
    return;
  }
  objs.push_back(obj);
}

inline void references::replace(changeable old, changeable newobj) {
  for (size_t i = 0; i < objs.size(); ++i) {
    if (objs[i] == old) {
      objs[i] = newobj;
    }
  }
}

inline void references::remove(changeable d) {
  for (size_t i = 0; i < objs.size(); ++i) {
    if (objs[i] == d) {
      objs.erase(objs.begin()+i);
      break;
    }
  }
}

