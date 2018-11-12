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
  using Function = std::function<void (FunctionArgs &)>;

  /**
   * @brief TypesFunction, that does the input types checking and output types derivation.
   *
   * The function is used within Entry::evaluateTypes() and Entry::updateTypes().
   *
   * @param TypesFunctionArgs -- container with transformation types (Atypes, Rtypes, Itypes).
   */
  using TypesFunction = std::function<void (TypesFunctionArgs &)>;

  using TypesFunctionsContainer = std::vector<TypesFunction>;

  /**
   * @brief StorageTypesFunction, that does the storage types derivation
   *
   * The function is used within Entry::evaluateTypes() and Entry::updateTypes().
   * Unlike the TypesFunction the StorageTypesFunction is not able to modify Rets (outputs).
   *
   * @param StorageTypesFunctionArgs -- container with transformation types (Atypes, Rtypes, Itypes).
   */
  using StorageTypesFunction = std::function<void (StorageTypesFunctionArgs &)>;

  using StorageTypesFunctionsContainer = std::vector<StorageTypesFunction>;
}
