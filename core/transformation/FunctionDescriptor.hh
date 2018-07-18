#pragma once

#include <map>
#include "TransformationFunction.hh"

namespace TransformationTypes
{
  struct Storage;
  typedef boost::ptr_vector<Storage> StoragesContainer;  ///< Container for Storage pointers.

  struct FunctionDescriptor {
    Function fun;
    TypesFunctionsContainer typefuns;
  };

  typedef std::map<std::string, FunctionDescriptor> FunctionDescriptorsContainer;
}
