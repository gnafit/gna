#include "TypesFunctions.hh"

using TransformationTypes::Atypes;
using TransformationTypes::Rtypes;

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
void TypesFunctions::passAll(Atypes args, Rtypes rets) {
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
void TypesFunctions::ifSame(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] != args[0]) {
      auto fmt = format("Transformation %1%: all inputs should have same type, %2% and %3% differ");
      throw args.error(args[i], (fmt%args.name()%0%i).str());
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
void TypesFunctions::ifSameShape(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i].shape != args[0].shape) {
      auto fmt = format("Transformation %1%: all inputs should have same shape, %2% and %3% differ");
      throw args.error(args[i], (fmt%args.name()%0%i).str());
    }
  }
}

