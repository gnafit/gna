#pragma once

#include <vector>
#include <algorithm>

class SimpleDictBase {
public:
  virtual ~SimpleDictBase() { }
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
    if (static_cast<size_t>(i) >= size()) {
      return T::invalid(i);
    }
    return at(i);
  }

  T back() const {
    return T(m_container->back());
  }

  T operator[](const std::string &name) const {
    auto it = std::find_if(m_container->begin(), m_container->end(),
                           [&](typename Container::const_reference e) {
                             return e.name == name;
                           });
    if (it == m_container->end()) {
      return T::invalid(name);
    }
    return *it;
  }
private:
  Container *m_container;
};
