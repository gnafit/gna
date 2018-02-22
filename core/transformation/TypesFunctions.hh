#pragma once

#include <string>
#include <boost/format.hpp>

#include "Atypes.hh"

struct TypesFunctions
{
  static void passAll(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);     ///< Assigns shape of each input to corresponding output.

  template <size_t Arg, size_t Ret = Arg>
  static void pass(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);        ///< Assigns shape of Arg-th input to Ret-th output.

  template <size_t Arg, size_t Ret = Arg>
  static void binsToEdges(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets); ///< Assigns shape of Arg-th input to Ret-th output with size=N+1.

  template <size_t Arg, size_t Ret = Arg>
  static void edgesToBins(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets); ///< Assigns shape of Arg-th input to Ret-th output with size=N-1.

  static void ifSame(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);      ///< Checks that all inputs are of the same type (shape and content description).
  static void ifSameShape(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets); ///< Checks that all inputs are of the same shape.

  template <size_t Arg1, size_t Arg2>
  static void ifSame2(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);      ///< Checks that inputs Arg1 and Arg2 are of the same type (shape and content description).
  template <size_t Arg1, size_t Arg2>
  static void ifSameShape2(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets); ///< Checks that inputs Arg1 and Arg2 inputs are of the same shape.

  template <size_t Arg>
  static void ifHist(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);      ///< Checks if Arg-th input is a histogram (DataKind=Histogram).

  template <size_t Arg>
  static void ifPoints(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);    ///< Checks if Arg-th input is an array (DataKind=Points).

  template <size_t Arg, size_t Ndim>
  static void ifNd(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);        ///< Checks if Arg-th input is N-dimensional.

  template <size_t Arg>
  static void if1d(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);        ///< Checks if Arg-th input is 1-dimensional.

  template <size_t Arg>
  static void if2d(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets);        ///< Checks if Arg-th input is 2-dimensional.

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
 * @exception SourceTypeError in case of invalid index is passed for args.
 * @exception SinkTypeError in case of invalid index is passed rets.
 */
template <size_t Arg, size_t Ret>
inline void TypesFunctions::pass(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (Arg >= args.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Arg index (%2% out of %3%)");
    throw args.error( DataType::undefined(), (fmt%args.name()%Arg%args.size()).str() );
  }
  if (Ret >= rets.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Ret index (%2% out of %3%)");
    throw rets.error(DataType::undefined(), (fmt%rets.name()%Ret%rets.size()).str() );
  }
  rets[Ret] = args[Arg];
}

/**
 * @brief Checks that inputs Arg1 and Arg2 are of the same type (shape and content description).
 *
 * Raises an exception otherwise.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input types are not the same.
 */
template <size_t Arg1, size_t Arg2>
void TypesFunctions::ifSame2(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (args[Arg1] != args[Arg2]) {
    auto fmt = format("Transformation %1%: inputs %2% and %3% should have same type");
    throw args.error(args[Arg2], (fmt%args.name()%Arg1%Arg2).str());
  }
}

/**
 * @brief Checks that inputs Arg1 and Arg2 are of the same shape.
 *
 * Raises an exception otherwise.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input shapes are not the same.
 */
template <size_t Arg1, size_t Arg2>
void TypesFunctions::ifSameShape2(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (args[Arg1].shape != args[Arg2].shape) {
    auto fmt = format("Transformation %1%: inputs %2% and %3% should have same shape");
    throw args.error(args[Arg2], (fmt%args.name()%Arg1%Arg2).str());
  }
}


/**
 * @brief Assigns shape of Arg-th input to Ret-th output. The ret-s size is N+1.
 *
 * @tparam Arg -- index of Arg to read the type.
 * @tparam Ret -- index of Ret to write the type (by default Ret=Arg)
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case of invalid index is passed for args.
 * @exception SinkTypeError in case of invalid index is passed rets.
 */
template <size_t Arg, size_t Ret>
inline void TypesFunctions::binsToEdges(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (Arg >= args.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Arg index (%2% out of %3%)");
    throw args.error( DataType::undefined(),  (fmt%args.name()%Arg%args.size()).str() );
  }
  if (Ret >= rets.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Ret index (%2% out of %3%)");
    throw rets.error( DataType::undefined(), (fmt%rets.name()%Ret%rets.size()).str() );
  }
  TypesFunctions::ifPoints<Arg>(args, rets);
  TypesFunctions::if1d<Arg>(args, rets);
  rets[Ret] = args[Arg];
  rets[Ret].shape[0]+=1;
}

/**
 * @brief Assigns shape of Arg-th input to Ret-th output. The ret-s size is N+1.
 *
 * @tparam Arg -- index of Arg to read the type.
 * @tparam Ret -- index of Ret to write the type (by default Ret=Arg)
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case of invalid index is passed for args.
 * @exception SinkTypeError in case of invalid index is passed rets.
 */
template <size_t Arg, size_t Ret>
inline void TypesFunctions::edgesToBins(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (Arg >= args.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Arg index (%2% out of %3%)");
    throw args.error( DataType::undefined(),  (fmt%args.name()%Arg%args.size()).str() );
  }
  if (Ret >= rets.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Ret index (%2% out of %3%)");
    throw rets.error( DataType::undefined(),  (fmt%rets.name()%Ret%rets.size()).str() );
  }
  TypesFunctions::ifPoints<Arg>(args, rets);
  TypesFunctions::if1d<Arg>(args, rets);
  rets[Ret] = args[Arg];
  rets[Ret].shape[0]-=1;
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
 * @exception SourceTypeError in case input data is not a histogram.
 */
template <size_t Arg>
inline void TypesFunctions::ifHist(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (args[Arg].kind!=DataKind::Hist) {
    auto fmt = boost::format("Transformation %1%: Arg %2% should be a histogram");
    throw args.error(args[Arg], (fmt%args.name()%Arg).str());
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
 * @exception SourceTypeError in case input data is not an array.
 */
template <size_t Arg>
inline void TypesFunctions::ifPoints(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  if (args[Arg].kind!=DataKind::Points) {
    auto fmt = boost::format("Transformation %1%: Arg %2% should be an array");
    throw args.error(args[Arg], (fmt%args.name()%Arg).str());
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
 * @exception SourceTypeError in case input data is not N-dimensional.
 */
template <size_t Arg, size_t Ndim>
inline void TypesFunctions::ifNd(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  auto ndim=args[Arg].shape.size();
  if (ndim!=Ndim) {
    auto fmt = boost::format("Transformation %1%: Arg %2% should be %3%-dimensional, not %4%-dimensional");
    throw args.error(args[Arg], (fmt%args.name()%Arg%Ndim%ndim).str());
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
 * @exception SourceTypeError in case input data is not 1-dimensional.
 */
template <size_t Arg>
inline void TypesFunctions::if1d(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  TypesFunctions::ifNd<Arg,1>(args, rets);
}


/**
 * @brief Checks if Arg-th input is 2d.
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input data is not 2-dimensional.
 */
template <size_t Arg>
inline void TypesFunctions::if2d(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  TypesFunctions::ifNd<Arg,2>(args, rets);
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
 * @exception SourceTypeError in case input data is not square (NxN).
 */
template <size_t Arg>
inline void TypesFunctions::ifSquare(TransformationTypes::Atypes args, TransformationTypes::Rtypes rets) {
  auto shape = args[Arg].shape;
  if (shape.size()!=2 || shape[0]!=shape[1] ) {
    auto fmt = boost::format("Transformation %1%: Arg %2% should be NxN, got %3%x%4%");
    throw args.error(args[Arg], (fmt%args.name()%Arg%shape[0]%shape[1]).str());
  }
}
