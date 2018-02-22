#pragma once

#include <string>
#include <boost/format.hpp>

#include "Atypes.hh"

struct TypesFunctions
{
  static void passAll(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);     ///< Assigns shape of each input to corresponding output.

  template <size_t Arg, size_t Ret = Arg>
  static void pass(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);        ///< Assigns shape of Arg-th input to Ret-th output.

  static void ifSame(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);      ///< Checks that all inputs are of the same type (shape and content description).
  static void ifSameShape(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets); ///< Checks that all inputs are of the same shape.

  template <size_t Arg>
  static void ifHist(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);      ///< Checks if Arg-th input is a histogram (DataKind=Histogram).

  template <size_t Arg>
  static void ifPoints(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);    ///< Checks if Arg-th input is an array (DataKind=Points).

  template <size_t Arg, size_t Ndim=1>
  static void ifNd(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);        ///< Checks if Arg-th input is N-dimensional (1 by default).

  template <size_t Arg>
  static void ifSquare(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);    ///< Checks if Arg-th input is of square shape.

};

/**
 * @brief Assigns shape of Arg-th input to Ret-th output
 *
 * @tparam Arg -- index of Arg to read the type.
 * @tparam Ret -- index of Ret to write the type (by default Ret=Arg)
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case of invalid index is passed.
 */
template <size_t Arg, size_t Ret>
inline void TypesFunctions::pass(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (Arg >= args.size()) {
    auto fmt = boost::format("Transformation: invalid Arg index (%1% out of %2%)");
    throw std::runtime_error( (fmt%Arg%args.size()).str() );
  }
  if (Ret >= rets.size()) {
    auto fmt = boost::format("Transformation: invalid Ret index (%1% out of %2%)");
    throw std::runtime_error( (fmt%Ret%rets.size()).str() );
  }
  rets[Ret] = args[Arg];
}

/**
 * @brief Checks if Arg-th input is a histogram (DataKind=Histogram).
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case input data is not a histogram.
 */
template <size_t Arg>
inline void TypesFunctions::ifHist(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (args[Arg].kind!=DataKind::Hist) {
    auto fmt = boost::format("Transformation: Arg %1% should be a histogram");
    throw std::runtime_error((fmt%Arg).str());
  }
}

/**
 * @brief Checks if Arg-th input is an array (DataKind=Points).
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case input data is not an array.
 */
template <size_t Arg>
inline void TypesFunctions::ifPoints(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (args[Arg].kind!=DataKind::Points) {
    auto fmt = boost::format("Transformation: Arg %1% should be an array");
    throw std::runtime_error((fmt%Arg).str());
  }
}

/**
 * @brief Checks if Arg-th input is 1d.
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case input data is not 1 dimensional.
 */
template <size_t Arg, size_t Ndim>
inline void TypesFunctions::ifNd(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  auto ndim=args[Arg].shape.size();
  if (ndim!=Ndim) {
    auto fmt = boost::format("Transformation: Arg %1% should be %2%-dimensional, not %3%-dimensional");
    throw std::runtime_error((fmt%Arg%Ndim%ndim).str());
  }
}

/**
 * @brief Checks if Arg-th input is of square shape
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case input data is not square (NxN).
 */
template <size_t Arg>
inline void TypesFunctions::ifSquare(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  auto shape = args[Arg].shape;
  if (shape.size()!=2 || shape[0]!=shape[1] ) {
    auto fmt = boost::format("Transformation: Arg %1% should be NxN, got %2%x%3%");
    throw std::runtime_error((fmt%Arg%shape[0]%shape[1]).str());
  }
}
