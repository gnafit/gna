#pragma once

#include "changeable.hh"

class callback: public changeable {
public:
  callback() { init(); }
  callback(std::function<void()> f, std::initializer_list<changeable> deps) {
    init(deps);
    m_data.hdr->on_taint = f;
  }
  template <typename T>
  callback(std::function<void()> f, std::vector<T> deps) {
    init(deps);
    m_data.hdr->on_taint = f;
  }
};
