#pragma once

#include "callback.hh"

class fragile: public callback {
public:
  fragile(const std::string& name="fragile") : callback(name) { init(); }

  fragile(changeable dep, const std::string& name="fragile") : callback(&fragile::on_taint, {dep}, name) { init(); }

  fragile(std::initializer_list<changeable> deps, const std::string& name="fragile") : callback(&fragile::on_taint, deps, name) { init(); }

  template <typename T>
  fragile(std::vector<T> deps, const std::string& name="fragile") : callback(&fragile::on_taint, deps, name) { init(); }

  static void on_taint() {
    throw std::runtime_error("fragile variable/transformation tainted");
  }

private:
  void init(){ m_hdr->tainted = false; }
};
