#pragma once

#include "TransformationFunction.hh"
//#include "Itypes.hh"

namespace TransformationTypes
{
  struct Storage;
  typedef boost::ptr_vector<Storage> StoragesContainer;  ///< Container for Storage pointers.

  struct FunctionDescriptor {
    Function* fun;

  };
}
