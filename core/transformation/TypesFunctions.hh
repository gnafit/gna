#pragma once

#include <string>
#include "fmt/format.h"

#include "TransformationFunctionArgs.hh"

using TransformationTypes::TypesFunctionArgsT;

struct TypesFunctions
{
  using TypesFunctionArgs = TransformationTypes::TypesFunctionArgsT<double,double>;

  template<typename FloatType>
  static void passAllT(TypesFunctionArgsT<FloatType,FloatType>& fargs);       ///< Assigns shape of each input to corresponding output.

  static void passAll(TypesFunctionArgs& fargs);       ///< Assigns shape of each input to corresponding output.

  template <int Arg1=0, int Arg2=-1, int Ret1=0>
  static void passAllInRange(TypesFunctionArgs& fargs); ///< Assigns shape of each input to corresponding output.

  template <int Arg=0, int Ret1=0, int Ret2=-1, bool ignore_bound_error=false>
  static void passToRange(TypesFunctionArgs& fargs);  ///< Assigns shape of each input to corresponding output.

  template <int Ret1=0, int Ret2=-1, bool ignore_bound_error=false>
  static void passNonSingle(TypesFunctionArgs& fargs); ///< Assigns shape of the first not 1x1 input to each output in the range.

  template <size_t Arg, size_t Ret = Arg>
  static void pass(TypesFunctionArgs& fargs);         ///< Assigns shape of Arg-th input to Ret-th output.

  template <size_t Ret>
  static void empty1(TypesFunctionArgs& fargs);       ///< Ret-th output shape {0}.

  template <size_t Ret>
  static void empty2(TypesFunctionArgs& fargs);       ///< Ret-th output shape {0x0}.

  template <size_t Arg, size_t Ret = Arg>
  static void binsToEdges(TypesFunctionArgs& fargs);  ///< Assigns shape of Arg-th input to Ret-th output with size=N+1.

  template <size_t Arg, size_t Ret = Arg>
  static void edgesToBins(TypesFunctionArgs& fargs);  ///< Assigns shape of Arg-th input to Ret-th output with size=N-1.

  template <size_t Arg1, size_t Arg2, size_t Ret>
  static void toMatrix(TypesFunctionArgs& fargs);     ///< Assigns shape of Ret-th output = [Arg1.size(), Arg2.size()] (ignoring Arg1/Arg2 shape)

  template <size_t Arg1, size_t Arg2, size_t Ret>
  static void edgesToMatrix(TypesFunctionArgs& fargs); ///< Assigns shape of Ret-th output = [Arg1.size()-1, Arg2.size()-1] (ignoring Arg1/Arg2 shape)

  template <int Arg1=0, int Arg2=-1, bool ignore_bound_error=false>
  static void ifSameInRange(TypesFunctionArgs& fargs); ///< Checks that all inputs are of the same type (shape and content description).

  static void ifSame(TypesFunctionArgs& fargs);       ///< Checks that all inputs are of the same type (shape and content description).
  static void ifSameShape(TypesFunctionArgs& fargs);  ///< Checks that all inputs are of the same shape.
  static void ifSameShapeOrSingle(TypesFunctionArgs& fargs);  ///< Checks that all inputs are of the same shape or 1x1

  template <size_t Arg1, size_t Arg2>
  static void ifSame2(TypesFunctionArgs& fargs);      ///< Checks that inputs Arg1 and Arg2 are of the same type (shape and content description).
  template <size_t Arg1, size_t Arg2>
  static void ifSameShape2(TypesFunctionArgs& fargs); ///< Checks that inputs Arg1 and Arg2 inputs are of the same shape.
  template <size_t Arg1, size_t Arg2>
  static void ifBinsEdges(TypesFunctionArgs& fargs);  ///< Checks that inputs Arg1 and Arg2 inputs has shape as bins and edges (N, N+1).

  template <size_t Arg>
  static void ifHist(TypesFunctionArgs& fargs);       ///< Checks if Arg-th input is a histogram (DataKind=Histogram).

  template <size_t Arg>
  static void ifPoints(TypesFunctionArgs& fargs);     ///< Checks if Arg-th input is an array (DataKind=Points).

  template <size_t Arg, size_t Ndim>
  static void ifNd(TypesFunctionArgs& fargs);         ///< Checks if Arg-th input is N-dimensional.

  template <size_t Arg>
  static void ifEmpty(TypesFunctionArgs& fargs);      ///< Checks if Arg-th input has zero size.

  template <size_t Arg>
  static void if1d(TypesFunctionArgs& fargs);         ///< Checks if Arg-th input is 1-dimensional.

  template <size_t Arg>
  static void if2d(TypesFunctionArgs& fargs);         ///< Checks if Arg-th input is 2-dimensional.

  template <size_t Arg>
  static void ifSquare(TypesFunctionArgs& fargs);     ///< Checks if Arg-th input is of square shape.
};

/**
 * @brief Assigns shape of Arg-th input to Ret-th output
 *
 * @tparam Arg -- index of Arg to read the type.
 * @tparam Ret -- index of Ret to write the type (by default Ret=Arg)
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case of invalid index is passed for args.
 * @exception SinkTypeError in case of invalid index is passed rets.
 */
template <size_t Arg, size_t Ret>
inline void TypesFunctions::pass(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (Arg >= args.size()) {
    auto msg = fmt::format("Transformation {0}: invalid Arg index ({1} out of {2})",args.name(), Arg, args.size());
    throw args.error( DataType::undefined(), msg );
  }
  if (Ret >= rets.size()) {
    auto msg = fmt::format("Transformation {0}: invalid Ret index ({1} out of {3})", rets.name(), Ret, rets.size());
    throw rets.error(DataType::undefined(), msg );
  }
  rets[Ret] = args[Arg];
}

/**
 * @brief Sets Ret-th output to shape 0.
 *
 * @tparam Ret -- index of Ret to write the type
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SinkTypeError in case of invalid index is passed for rets.
 */
template <size_t Ret>
inline void TypesFunctions::empty1(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& rets=fargs.rets;
  if (Ret >= rets.size()) {
    auto msg = fmt::format("Transformation {0}: invalid Ret index ({1} out of {2})", rets.name(), Ret, rets.size());
    throw rets.error(DataType::undefined(), msg );
  }
  rets[Ret] = DataType().points().shape(0);
}

/**
 * @brief Sets Ret-th output to shape 0x0.
 *
 * @tparam Ret -- index of Ret to write the type
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SinkTypeError in case of invalid index is passed for rets.
 */
template <size_t Ret>
inline void TypesFunctions::empty2(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& rets=fargs.rets;
  if (Ret >= rets.size()) {
    auto msg = fmt::format("Transformation {0}: invalid Ret index ({1} out of {2})", rets.name(), Ret, rets.size());
    throw rets.error(DataType::undefined(), msg );
  }
  rets[Ret] = DataType().points().shape(0,0);
}

/**
 * @brief Assigns shape of Ret-th output = [Arg1.size(), Arg2.size()] (ignoring Arg1/Arg2 shape)
 *
 * @tparam Arg1 -- index of Arg1 to read the size.
 * @tparam Arg2 -- index of Arg2 to read the size.
 * @tparam Ret -- index of Ret to write the type.
 *
 * @param fargs -- input/output types.
 */
template <size_t Arg1, size_t Arg2, size_t Ret>
inline void TypesFunctions::toMatrix(TypesFunctions::TypesFunctionArgs& fargs) {
  fargs.rets[Ret] = DataType().points().shape(fargs.args[Arg1].size(), fargs.args[Arg2].size());
}

/**
 * @brief Assigns shape of Ret-th output = [Arg1.size()-1, Arg2.size()-1] (ignoring Arg1/Arg2 shape)
 *
 * @tparam Arg1 -- index of Arg1 to read the size.
 * @tparam Arg2 -- index of Arg2 to read the size.
 * @tparam Ret -- index of Ret to write the type.
 *
 * @param fargs -- input/output types.
 */
template <size_t Arg1, size_t Arg2, size_t Ret>
inline void TypesFunctions::edgesToMatrix(TypesFunctions::TypesFunctionArgs& fargs) {
  fargs.rets[Ret] = DataType().points().shape(fargs.args[Arg1].size()-1, fargs.args[Arg2].size()-1);
}


/**
 * @brief Checks that all inputs in a range are of the same type (shape and content description).
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg1 -- index of Arg1 to start comparison from.
 * @tparam Arg2 -- index of Arg2 to end comparison (inclusive).
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input types are not the same.
 */
template <int Arg1, int Arg2, bool ignore_bound_error>
void TypesFunctions::ifSameInRange(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& compare_to=args[Arg1];
  size_t start=Arg1+1;
  size_t end=Arg2<0 ? args.size()+Arg2+1 : Arg2+1;

  if( start>=args.size() || end>args.size() ){
    if(ignore_bound_error){
      return;
    }
    else{
      throw std::runtime_error(fmt::format("Transformation {0}: start {1} or end {2} is out of limits ({3})", args.name(),  start, end, args.size() ));
    }
  }

  for (size_t i = start; i < end; ++i) {
    if (args[i] != compare_to) {
      auto msg = fmt::format("Transformation {0}: all inputs should have same type, {1} and {2} differ", args.name(), Arg1, i);
      throw args.error(args[i], msg);
    }
  }
}

/**
 * @brief Checks that inputs Arg1 and Arg2 are of the same type (shape and content description).
 *
 * Raises an exception otherwise.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input types are not the same.
 */
template <size_t Arg1, size_t Arg2>
void TypesFunctions::ifSame2(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  if (args[Arg1] != args[Arg2]) {
    auto msg = fmt::format("Transformation {0}: inputs {1} and {2} should have same type", args.name(), Arg1, Arg2);
    throw args.error(args[Arg2], msg);
  }
}

/**
 * @brief Checks that inputs Arg1 and Arg2 are of the same shape.
 *
 * Raises an exception otherwise.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input shapes are not the same.
 */
template <size_t Arg1, size_t Arg2>
void TypesFunctions::ifSameShape2(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  if (args[Arg1].shape != args[Arg2].shape) {
    auto msg = fmt::format("Transformation {0}: inputs {1} and {2} should have same shape", args.name(), Arg1, Arg2);
    throw args.error(args[Arg2], msg);
  }
}

/**
 * @brief Checks that inputs Arg1 and Arg2 are bins and edges with N and N+1 elements.
 *
 * Raises an exception otherwise.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input shapes are not the same.
 */
template <size_t Arg1, size_t Arg2>
void TypesFunctions::ifBinsEdges(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  TypesFunctions::if1d<Arg1>(fargs);
  TypesFunctions::if1d<Arg2>(fargs);
  if (args[Arg1].shape[0] != (args[Arg2].shape[0]-1u)) {
    auto msg = fmt::format("Transformation {0}: inputs {1} and {2} should sizes N and N+1, got {3} and {4}",
               args.name(), Arg1, Arg2, args[Arg1].shape[0], args[Arg2].shape[0]);
    throw args.error(args[Arg2], msg);
  }
}

/**
 * @brief Assigns shape of Arg-th input to Ret-th output. The ret-s size is N+1.
 *
 * @tparam Arg -- index of Arg to read the type.
 * @tparam Ret -- index of Ret to write the type (by default Ret=Arg)
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case of invalid index is passed for args.
 * @exception SinkTypeError in case of invalid index is passed rets.
 */
template <size_t Arg, size_t Ret>
inline void TypesFunctions::binsToEdges(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (Arg >= args.size()) {
    auto msg = fmt::format("Transformation {0}: invalid Arg index ({1} out of {2})", args.name(), Arg, args.size());
    throw args.error( DataType::undefined(), msg );
  }
  if (Ret >= rets.size()) {
    auto msg = fmt::format("Transformation {0}: invalid Ret index ({1} out of {2})", rets.name(), Ret, rets.size());
    throw rets.error( DataType::undefined(), msg );
  }
  TypesFunctions::if1d<Arg>(fargs);
  rets[Ret] = args[Arg];
  rets[Ret].shape[0]+=1;
}

/**
 * @brief Assigns shape of Arg-th input to Ret-th output. The ret-s size is N+1.
 *
 * @tparam Arg -- index of Arg to read the type.
 * @tparam Ret -- index of Ret to write the type (by default Ret=Arg)
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case of invalid index is passed for args.
 * @exception SinkTypeError in case of invalid index is passed rets.
 */
template <size_t Arg, size_t Ret>
inline void TypesFunctions::edgesToBins(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (Arg >= args.size()) {
    auto msg = fmt::format("Transformation {0}: invalid Arg index ({1} out of {2})");
    throw args.error( DataType::undefined(),  msg );
  }
  if (Ret >= rets.size()) {
    auto msg = fmt::format("Transformation {0}: invalid Ret index ({1} out of {2})", rets.name(), Ret, rets.size());
    throw rets.error( DataType::undefined(),  msg );
  }
  TypesFunctions::ifPoints<Arg>(fargs);
  TypesFunctions::if1d<Arg>(fargs);
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
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not a histogram.
 */
template <size_t Arg>
inline void TypesFunctions::ifHist(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  if (args[Arg].kind!=DataKind::Hist) {
    auto msg = fmt::format("Transformation {0}: Arg {1} should be a histogram", args.name(), Arg);
    throw args.error(args[Arg], msg);
  }
}

/**
 * @brief Checks if Arg-th input is an array (DataKind=Points).
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not an array.
 */
template <size_t Arg>
inline void TypesFunctions::ifPoints(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  if (args[Arg].kind!=DataKind::Points) {
    auto msg = fmt::format("Transformation {0}: Arg {1} should be an array", args.name(), Arg);
    throw args.error(args[Arg], msg);
  }
}

/**
 * @brief Checks if Arg-th input is 1d.
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not N-dimensional.
 */
template <size_t Arg, size_t Ndim>
inline void TypesFunctions::ifNd(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto ndim=args[Arg].shape.size();
  if (ndim!=Ndim) {
    auto msg = fmt::format("Transformation {0}: Arg {1} should be {2}-dimensional, not {3}-dimensional", args.name(), Arg, Ndim, ndim);
    throw args.error(args[Arg], msg);
  }
}

/**
 * @brief Checks if Arg-th input is 1d.
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not 1-dimensional.
 */
template <size_t Arg>
inline void TypesFunctions::if1d(TypesFunctions::TypesFunctionArgs& fargs) {
  TypesFunctions::ifNd<Arg,1>(fargs);
}

/**
 * @brief Checks if Arg-th input has 0 size.
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not 1-dimensional.
 */
template <size_t Arg>
inline void TypesFunctions::ifEmpty(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto size=args[Arg].size();
  if (size!=0u) {
    auto msg = fmt::format("Transformation {0}: Arg {1} should has 0 elements", args.name(), Arg);
    throw args.error(args[Arg], msg);
  }
}

/**
 * @brief Checks if Arg-th input is 2d.
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not 2-dimensional.
 */
template <size_t Arg>
inline void TypesFunctions::if2d(TypesFunctions::TypesFunctionArgs& fargs) {
  TypesFunctions::ifNd<Arg,2>(fargs);
}
/**
 * @brief Checks if Arg-th input is of square shape
 *
 * Raises an exception otherwise.
 *
 * @tparam Arg -- index of Arg to check.
 *
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not square (NxN).
 */
template <size_t Arg>
inline void TypesFunctions::ifSquare(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto shape = args[Arg].shape;
  if (shape.size()!=2 || shape[0]!=shape[1] ) {
    auto msg = fmt::format("Transformation {0}: Arg {1} should be NxN, got {2}x{3}",args.name(), Arg, shape[0], shape[1]);
    throw args.error(args[Arg], msg);
  }
}

/**
 * @brief Assigns shape of each input to corresponding output.
 *
 * In case of single input and multiple outputs assign its size to each output.
 *
 * @tparam Arg1 -- index of Arg to start comparison with.
 * @tparam Arg2 -- index of Arg to stop comparison with (inclusive).
 * @tparam Ret1 -- index of Ret to start writing to.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case Ret index is out of range.
 */
template <int Arg1, int Arg2, int Ret1>
inline void TypesFunctions::passAllInRange(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  size_t start=Arg1<0 ? args.size()+Arg1   : Arg1;
  size_t end  =Arg2<0 ? args.size()+Arg2+1 : Arg2+1;
  size_t ret  =Ret1<0 ? rets.size()+Ret1   : Ret1;
  for (size_t i = start; i < end; ++i) {
    if(ret>=rets.size()){
      auto msg = fmt::format("Transformation {0}: ret {1} is out of limits ({2})", rets.name(), ret, rets.size() );
      throw std::runtime_error(msg);
    }
    rets[ret] = args[i];
    ++ret;
  }
}

/**
 * @brief Assigns shape of input to each of outputs in a range.
 *
 * In case of single input and multiple outputs assign its size to each output.
 *
 * @tparam Arg1 -- index of Arg to pass;
 * @tparam Ret1 -- index of Ret to start writing to.
 * @tparam Ret2 -- index of Ret to stop writing to (inclusive).
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case Ret index is out of range.
 */
template <int Arg1, int Ret1, int Ret2, bool ignore_bound_error>
inline void TypesFunctions::passToRange(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  size_t arg  =Arg1<0 ? args.size()+Arg1   : Arg1;
  size_t start=Ret1<0 ? rets.size()+Ret1   : Ret1;
  size_t end  =Ret2<0 ? rets.size()+Ret2+1 : Ret2+1;

  if(arg>=args.size()){
    if(ignore_bound_error){
      return;
    }
    else{
      throw std::runtime_error(fmt::format("Transformation {0}: arg {1} is out of limits ({2})", rets.name(), arg, args.size() ));
    }
  }

  for (size_t i = start; i < end; ++i) {
    if(i>=rets.size()){
      if(ignore_bound_error){
        return;
      }
      else{
        auto msg = fmt::format("Transformation {0}: ret {1} is out of limits ({2})", rets.name(), i, rets.size() );
        throw std::runtime_error(msg);
      }
    }
    rets[i] = args[arg];
  }
}

/**
 * @brief Assigns shape of first not 1x1 input to each of the outputs in a range.
 *
 * In case all the inputs are 1x1, assign 1x1
 *
 * @tparam Ret1 -- index of Ret to start writing to.
 * @tparam Ret2 -- index of Ret to stop writing to (inclusive).
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case Ret index is out of range.
 */
template <int Ret1, int Ret2, bool ignore_bound_error>
inline void TypesFunctions::passNonSingle(TypesFunctions::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  DataType dt = DataType().points().shape(1);
  for (size_t i = 0; i < args.size(); ++i) {
    if(args[i].shape!=dt.shape){
      dt=args[i];
      break;
    }
  }

  size_t start=Ret1<0 ? rets.size()+Ret1   : Ret1;
  size_t end  =Ret2<0 ? rets.size()+Ret2+1 : Ret2+1;
  for (size_t i = start; i < end; ++i) {
    if(i>=rets.size()){
      if(ignore_bound_error){
        return;
      }
      else{
        auto msg = fmt::format("Transformation {0}: ret {1} is out of limits ({2})", rets.name(), i, rets.size() );
        throw std::runtime_error(msg);
      }
    }
    rets[i] = dt;
  }
}
