#pragma once

#include <map>
#include "TransformationFunction.hh"

namespace TransformationTypes
{
  template<typename FloatType> struct StorageT;

  template<typename FloatType>
  using StoragesContainerT = boost::ptr_vector<StorageT<FloatType>>;  ///< Container for Storage pointers.

  /**
   * @brief A class to keep transformation function information
   *
   * The Entry may have several functions implementations that may differ by:
   *   - The device: CPU or GPU
   *   - The particular device: GPU0, GPU1 etc
   *   - Requrements to the preallocated memory to be used by the transformation
   *
   * The memory allocation is in particular improtant for the GPU case, since it is quite a slow procedure.
   * Therefore the memory allocation is executed prior the computations on the binding stage.
   *
   * @author Maxim Gonchar
   * @date 18.07.2018
   */
  template<typename SourceFloatType, typename SinkFloatType>
  struct FunctionDescriptorT {
    FunctionT<SourceFloatType,SinkFloatType>                       fun;      ///< The pointer to the transformation Function
    StorageTypesFunctionsContainerT<SourceFloatType,SinkFloatType> typefuns; ///< Container with TypesFunction specifying the storage requirements for this particular function
  };

  template<typename SourceFloatType, typename SinkFloatType>
  using FunctionDescriptorsContainerT = std::map<std::string, FunctionDescriptorT<SourceFloatType,SinkFloatType>>;
}
