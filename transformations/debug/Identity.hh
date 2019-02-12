#pragma once

#include <iostream>
#include "GNAObject.hh"
#include "TypesFunctions.hh"

//
// Identity transformation
//
class Identity: public GNASingleObject,
                public TransformationBind<Identity> {
public:
  Identity();
  void dump();
};
