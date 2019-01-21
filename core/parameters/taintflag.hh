#pragma once

#include <iostream>
#include <vector>
#include "changeable.hh"

class taintflag: public changeable {
public:
  taintflag(const char* name="", bool autoname=false) { init(name, autoname); }
  taintflag(std::initializer_list<changeable> deps, const char* name="") { init(deps, name); }
  taintflag(std::vector<changeable> deps, const char* name="") { init(deps, name); }
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

  void set_pass_through(){
    m_data.hdr->status=TaintStatus::PassThrough;
  }
};

