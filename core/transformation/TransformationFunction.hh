#pragma once

#include <boost/ptr_container/ptr_vector.hpp>
#include <vector>

namespace TransformationTypes
{
  struct TypesFunctionArgs;
  struct FunctionArgs;

  /**
   * @brief Function, that does the actual calculation.
   *
   * This function is used to define the transformation via Entry::fun
   * and is executed via Entry::update() or Entry::touch().
   *
   * @param FunctionArgs -- container with transformation inputs (Args), outputs (Rets) and other data.
   */
  typedef std::function<void(FunctionArgs)> Function;

  /**
   * @brief Function, that does the input types checking and output types derivation.
   *
   * The function is used within Entry::evaluateTypes() and Entry::updateTypes().
   *
   * @param TypesFunctionArgs -- container with transformation types (Atypes, Rtypes).
   */
  typedef std::function<void(TypesFunctionArgs)> TypesFunction;

  typedef std::vector<TypesFunction> TypesFunctionsContainer;
}
