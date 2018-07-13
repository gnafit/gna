#pragma once

#include <boost/ptr_container/ptr_vector.hpp>
#include "Source.hh"
#include "Sink.hh"

namespace TransformationTypes
{
  struct FunctionArgs;
  struct FunctionArgs;
  struct Atypes;
  struct Rtypes;

  typedef boost::ptr_vector<Source> SourcesContainer;   ///< Container for Source pointers.
  typedef boost::ptr_vector<Sink>   SinksContainer;     ///< Container for Sink pointers.

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
   * @param atypes -- container with transformation inputs' types (Atypes).
   * @param rtypes -- container with transformation outputs' types (Rtypes).
   */
  typedef std::function<void(Atypes, Rtypes)> TypesFunction;
}
