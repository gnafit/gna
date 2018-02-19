#include "TransformationBase.hh"

#include <boost/format.hpp>
using boost::format;

using TransformationTypes::Atypes;

/**
 * @brief Assigns shape of each input to corresponding output.
 *
 * In case of single input and multiple outputs assign its size to each output.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case the number of inputs and outputs is >1 and not the same.
 */
void Atypes::passAll(Atypes args, Rtypes rets) {
  if (args.size() == 1) {
    for (size_t i = 0; i < rets.size(); ++i) {
      rets[i] = args[0];
    }
  } else if (args.size() != rets.size()) {
    auto fmt = format("Transformation %1%: nargs != nrets");
    throw std::runtime_error((fmt % args.name()).str());
  } else {
    for (size_t i = 0; i < args.size(); ++i) {
      rets[i] = args[i];
    }
  }
}

/**
 * @brief Checks that all inputs are of the same type (shape and content description).
 *
 * Raises an exception otherwise.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input types are not the same.
 */
void Atypes::ifSame(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] != args[0]) {
      throw args.error(args[i], "inputs should have same type");
    }
  }
}

/**
 * @brief Checks that all inputs are of the same shape.
 *
 * Raises an exception otherwise.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input shapes are not the same.
 */
void Atypes::ifSameShape(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i].shape != args[0].shape) {
      throw args.error(args[i], "inputs should have same shape");
    }
  }
}

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
inline void Atypes::pass(Atypes args, Rtypes rets) {
  if (Arg >= args.size()) {
	throw std::runtime_error("Transformation: invalid Arg index");
  }
  if (Ret >= rets.size()) {
	throw std::runtime_error("Transformation: invalid Ret index");
  }
  rets[Ret] = args[Arg];
}

/**
 * @brief Checks if Arg-th input is a histogram (DataKind=Histogram).
 *
 * Raises an exception otherwise.
 *
 *  @tparam Arg -- index of Arg to check.
 *
 *  @param args -- source types.
 *  @param rets -- output types.
 *
 *  @exception std::runtime_error in case input data is not a histogram.
 */
template <size_t Arg>
inline void Atypes::ifHist(Atypes args, Rtypes rets) {
  if (args[Arg].kind!=DataKind::Hist) {
	throw std::runtime_error("Transformation: Arg should be a histogram");
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
 *  @exception std::runtime_error in case input data is not an array.
 */
template <size_t Arg>
inline void Atypes::ifPoints(Atypes args, Rtypes rets) {
  if (args[Arg].kind!=DataKind::Points) {
	throw std::runtime_error("Transformation: Arg should be an array");
  }
}
