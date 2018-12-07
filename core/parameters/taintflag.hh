#pragma once

#include <iostream>
#include "changeable.hh"

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

  taintflag set(bool value){
    *this=value;
    return *this;
  }

  friend std::ostream& operator<< (std::ostream& out, const taintflag& tf) {
      if (tf) {
          out << tf.name() << " is tainted ";
      } else {
          out << tf.name() << " not tainted ";
      }
      return out;
  };
};

