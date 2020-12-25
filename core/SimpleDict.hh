#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>

struct SimpleDictBase {
public:
  virtual ~SimpleDictBase() = default;
};

template <typename T, typename Container>
struct SimpleDict: public SimpleDictBase {
public:
  SimpleDict(Container &container): m_container(&container) { }

  size_t size() const noexcept {
    return m_container->size();
  }

  T at(int i) const {
    if(i<0){
      i = static_cast<int>(size())>+i;
    }
    if(i<0){
      throw std::out_of_range("Negative index too 'large'");
    }
    return T(m_container->at(i));
  }

  T operator[](int i) const {
    if(i<0){
      i += static_cast<int>(size());
    }
    if (i<0 || static_cast<size_t>(i)>=size()) {
      return T::invalid(i);
    }
    return T(m_container->at(i));
  }

  T back() const {
    return T(m_container->back());
  }

  T front() const {
    return T(m_container->front());
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

  bool contains(const std::string &name) const {
    auto it = std::find_if(m_container->begin(), m_container->end(),
                           [&](typename Container::const_reference e) {
                             return e.name == name;
                           });
    if (it == m_container->end()) {
      return false;
    }
    return true;
  }

  bool empty() const noexcept {
    return m_container->empty();
  }
private:
  Container *m_container;
};
