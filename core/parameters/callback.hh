#pragma once

#include "changeable.hh"

class callback: public changeable {
public:
  callback(const std::string& name="callback") { init(name.c_str(), true); }
  callback(std::function<void()> f, std::initializer_list<changeable> deps, const std::string& name="callback") {
    init(deps, name.c_str());
    m_hdr->on_taint = f;
  }
  template <typename T>
  callback(std::function<void()> f, std::vector<T> deps, const std::string& name="callback") {
    init(deps, name.c_str());
    m_hdr->on_taint = f;
  }
};
