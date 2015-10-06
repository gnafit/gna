#ifndef SIMPLEDICT_H
#define SIMPLEDICT_H

#include <vector>
#include <algorithm>

#include "TObject.h"

class SimpleDictBase: public TObject {
};

template <typename T, typename Container>
class SimpleDict: public SimpleDictBase {
public:
  SimpleDict(Container &container): m_container(&container) { }

  size_t size() const {
    return m_container->size();
  }

  T at(int i) const {
    return T(m_container->at(i));
  }
  T operator[](int i) const {
    if (i >= size()) {
      return T::invalid(i);
    }
    return at(i);
  }
  T operator[](const std::string &name) const {
    auto it = std::find_if(m_container->begin(), m_container->end(),
                           [&](typename Container::const_reference e) {
                             return e.name == name;
                           });
    if (it == m_container->end()) {
      return T::invalid(name);
    }
    return T(*it);
  }
private:
  Container *m_container;

  ClassDef(SimpleDict, 0);
};

#endif // SIMPLEDICT_H
