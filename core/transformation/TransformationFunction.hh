#pragma once

#include <boost/ptr_container/ptr_vector.hpp>
#include <vector>

namespace TransformationTypes
{
  struct FunctionArgs;
  struct TypesFunctionArgs;
  struct StorageTypesFunctionArgs;

  /**
   * @brief Function, that does the actual calculation.
   *
   * This function is used to define the transformation via Entry::fun
   * and is executed via Entry::update() or Entry::touch().
   *
   * @param FunctionArgs -- container with transformation inputs (Args), outputs (Rets) and storages (Ints).
   */
  typedef std::function<void(FunctionArgs&)> Function;

  /**
   * @brief TypesFunction, that does the input types checking and output types derivation.
   *
   * The function is used within Entry::evaluateTypes() and Entry::updateTypes().
   *
   * @param TypesFunctionArgs -- container with transformation types (Atypes, Rtypes, Itypes).
   */
  typedef std::function<void(TypesFunctionArgs&)> TypesFunction;

  typedef std::vector<TypesFunction> TypesFunctionsContainer;

  /**
   * @brief StorageTypesFunction, that does the storage types derivation
   *
   * The function is used within Entry::evaluateTypes() and Entry::updateTypes().
   * Unlike the TypesFunction the StorageTypesFunction is not able to modify Rets (outputs).
   *
   * @param StorageTypesFunctionArgs -- container with transformation types (Atypes, Rtypes, Itypes).
   */
  typedef std::function<void(StorageTypesFunctionArgs&)> StorageTypesFunction;

  typedef std::vector<StorageTypesFunction> StorageTypesFunctionsContainer;
}
