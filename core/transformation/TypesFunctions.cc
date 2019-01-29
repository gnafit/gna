#include "TypesFunctions.hh"

using TransformationTypes::TypesFunctionArgs;

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
void TypesFunctions::passAll(TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (args.size() == 1) {
    for (size_t i = 0; i < rets.size(); ++i) {
      rets[i] = args[0];
    }
  } else if (args.size() != rets.size()) {
    auto msg = fmt::format("Transformation {0}: nargs != nrets", args.name());
    throw std::runtime_error(msg);
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
void TypesFunctions::ifSame(TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] != args[0]) {
      auto msg = fmt::format("Transformation {0}: all inputs should have same type, {1} and {2} differ", args.name(), 0, i);
      throw args.error(args[i], msg);
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
void TypesFunctions::ifSameShape(TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i].shape != args[0].shape) {
      auto msg = fmt::format("Transformation {0}: all inputs should have same shape, {1} and {2} differ", args.name(), 0, i);
      throw args.error(args[i], msg);
    }
  }
}


/**
 * @brief Checks that all inputs are of the same shape are 1x1.
 *
 * Raises an exception otherwise.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input shapes are not the same.
 */
void TypesFunctions::ifSameShapeOrSingle(TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  DataType dtsingle = DataType().points().shape(1);
  DataType dt = dtsingle;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].shape == dtsingle.shape) {
      continue;
    }
    if (dt.shape!=dtsingle.shape && args[i].shape!=dt.shape) {
      auto msg = fmt::format("Transformation {0}: all inputs should have same shape or be of dimension 1, error on input {1}", args.name(), i);
      printf("Current data type: ");
      args[i].dump();
      throw args.error(args[i], msg);
    }
    dt = args[i];
  }
  fargs.rets[0] = dt;
}
