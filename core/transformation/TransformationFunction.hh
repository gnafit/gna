#pragma once

#include <boost/ptr_container/ptr_vector.hpp>
#include <vector>

namespace TransformationTypes
{
  template<typename SourceFloatType, typename SinkFloatType>
  class FunctionArgsT;

  template<typename SourceFloatType, typename SinkFloatType>
  struct TypesFunctionArgsT;

  template<typename SourceFloatType, typename SinkFloatType>
  struct StorageTypesFunctionArgsT;

  /**
   * @brief Function, that does the actual calculation.
   *
   * This function is used to define the transformation via Entry::fun
   * and is executed via Entry::update() or Entry::touch().
   *
   * @param FunctionArgs -- container with transformation inputs (Args), outputs (Rets) and storages (Ints).
   */
  template<typename SourceFloatType, typename SinkFloatType>
  using FunctionT = std::function<void (FunctionArgsT<SourceFloatType,SinkFloatType>&)>;

  /**
   * @brief TypesFunction, that does the input types checking and output types derivation.
   *
   * The function is used within Entry::updateTypes().
   *
   * @param TypesFunctionArgs -- container with transformation types (Atypes, Rtypes, Itypes).
   */
  template<typename SourceFloatType, typename SinkFloatType>
  using TypesFunctionT = std::function<void (TypesFunctionArgsT<SourceFloatType,SinkFloatType> &)>;

  template<typename SourceFloatType, typename SinkFloatType>
  using TypesFunctionsContainerT = std::vector<TypesFunctionT<SourceFloatType,SinkFloatType>>;

  /**
   * @brief StorageTypesFunction, that does the storage types derivation
   *
   * The function is used within Entry::updateTypes().
   * Unlike the TypesFunction the StorageTypesFunction is not able to modify Rets (outputs).
   *
   * @param StorageTypesFunctionArgs -- container with transformation types (Atypes, Rtypes, Itypes).
   */
  template<typename SourceFloatType, typename SinkFloatType>
  using StorageTypesFunctionT = std::function<void (StorageTypesFunctionArgsT<SourceFloatType,SinkFloatType> &)>;

  template<typename SourceFloatType, typename SinkFloatType>
  using StorageTypesFunctionsContainerT = std::vector<StorageTypesFunctionT<SourceFloatType,SinkFloatType>>;
}
