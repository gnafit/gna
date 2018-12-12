#pragma once

#include <string>
#include <boost/format.hpp>

#include "TransformationFunctionArgs.hh"

struct TypesFunctions
{
  static void passAll(TransformationTypes::TypesFunctionArgs& fargs);       ///< Assigns shape of each input to corresponding output.

  template <int Arg1=0, int Arg2=-1, int Ret1=0>
  static void passAllInRange(TransformationTypes::TypesFunctionArgs& fargs); ///< Assigns shape of each input to corresponding output.

  template <int Arg=0, int Ret1=0, int Ret2=-1>
  static void passToRange(TransformationTypes::TypesFunctionArgs& fargs);  ///< Assigns shape of each input to corresponding output.

  template <size_t Arg, size_t Ret = Arg>
  static void pass(TransformationTypes::TypesFunctionArgs& fargs);         ///< Assigns shape of Arg-th input to Ret-th output.

  template <size_t Ret>
  static void empty1(TransformationTypes::TypesFunctionArgs& fargs);       ///< Ret-th output shape {0}.

  template <size_t Ret>
  static void empty2(TransformationTypes::TypesFunctionArgs& fargs);       ///< Ret-th output shape {0x0}.

  template <size_t Arg, size_t Ret = Arg>
  static void binsToEdges(TransformationTypes::TypesFunctionArgs& fargs);  ///< Assigns shape of Arg-th input to Ret-th output with size=N+1.

  template <size_t Arg, size_t Ret = Arg>
  static void edgesToBins(TransformationTypes::TypesFunctionArgs& fargs);  ///< Assigns shape of Arg-th input to Ret-th output with size=N-1.

  template <size_t Arg1, size_t Arg2, size_t Ret>
  static void toMatrix(TransformationTypes::TypesFunctionArgs& fargs);     ///< Assigns shape of Ret-th output = [Arg1.size(), Arg2.size()] (ignoring Arg1/Arg2 shape)

  template <size_t Arg1, size_t Arg2, size_t Ret>
  static void edgesToMatrix(TransformationTypes::TypesFunctionArgs& fargs); ///< Assigns shape of Ret-th output = [Arg1.size()-1, Arg2.size()-1] (ignoring Arg1/Arg2 shape)

  template <int Arg1=0, int Arg2=-1>
  static void ifSameInRange(TransformationTypes::TypesFunctionArgs& fargs); ///< Checks that all inputs are of the same type (shape and content description).

  static void ifSame(TransformationTypes::TypesFunctionArgs& fargs);       ///< Checks that all inputs are of the same type (shape and content description).
  static void ifSameShape(TransformationTypes::TypesFunctionArgs& fargs);  ///< Checks that all inputs are of the same shape.

  template <size_t Arg1, size_t Arg2>
  static void ifSame2(TransformationTypes::TypesFunctionArgs& fargs);      ///< Checks that inputs Arg1 and Arg2 are of the same type (shape and content description).
  template <size_t Arg1, size_t Arg2>
  static void ifSameShape2(TransformationTypes::TypesFunctionArgs& fargs); ///< Checks that inputs Arg1 and Arg2 inputs are of the same shape.
  template <size_t Arg1, size_t Arg2>
  static void ifBinsEdges(TransformationTypes::TypesFunctionArgs& fargs);  ///< Checks that inputs Arg1 and Arg2 inputs has shape as bins and edges (N, N+1).

  template <size_t Arg>
  static void ifHist(TransformationTypes::TypesFunctionArgs& fargs);       ///< Checks if Arg-th input is a histogram (DataKind=Histogram).

  template <size_t Arg>
  static void ifPoints(TransformationTypes::TypesFunctionArgs& fargs);     ///< Checks if Arg-th input is an array (DataKind=Points).

  template <size_t Arg, size_t Ndim>
  static void ifNd(TransformationTypes::TypesFunctionArgs& fargs);         ///< Checks if Arg-th input is N-dimensional.

  template <size_t Arg>
  static void ifEmpty(TransformationTypes::TypesFunctionArgs& fargs);      ///< Checks if Arg-th input has zero size.

  template <size_t Arg>
  static void if1d(TransformationTypes::TypesFunctionArgs& fargs);         ///< Checks if Arg-th input is 1-dimensional.

  template <size_t Arg>
  static void if2d(TransformationTypes::TypesFunctionArgs& fargs);         ///< Checks if Arg-th input is 2-dimensional.

  template <size_t Arg>
  static void ifSquare(TransformationTypes::TypesFunctionArgs& fargs);     ///< Checks if Arg-th input is of square shape.
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
inline void TypesFunctions::pass(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
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
inline void TypesFunctions::empty1(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& rets=fargs.rets;
  if (Ret >= rets.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Ret index (%2% out of %3%)");
    throw rets.error(DataType::undefined(), (fmt%rets.name()%Ret%rets.size()).str() );
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
inline void TypesFunctions::empty2(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& rets=fargs.rets;
  if (Ret >= rets.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Ret index (%2% out of %3%)");
    throw rets.error(DataType::undefined(), (fmt%rets.name()%Ret%rets.size()).str() );
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
inline void TypesFunctions::toMatrix(TransformationTypes::TypesFunctionArgs& fargs) {
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
inline void TypesFunctions::edgesToMatrix(TransformationTypes::TypesFunctionArgs& fargs) {
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
template <int Arg1=0, int Arg2=-1>
void TypesFunctions::ifSameInRange(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& compare_to=args[Arg1];
  size_t start=Arg1+1;
  size_t end=Arg2<0 ? args.size()+Arg2+1 : Arg2+1;
  for (size_t i = start; i < end; ++i) {
    if (args[i] != compare_to) {
      auto fmt = format("Transformation %1%: all inputs should have same type, %2% and %3% differ");
      throw args.error(args[i], (fmt%args.name()%0%i).str());
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
void TypesFunctions::ifSame2(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
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
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input shapes are not the same.
 */
template <size_t Arg1, size_t Arg2>
void TypesFunctions::ifSameShape2(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  if (args[Arg1].shape != args[Arg2].shape) {
    auto fmt = format("Transformation %1%: inputs %2% and %3% should have same shape");
    throw args.error(args[Arg2], (fmt%args.name()%Arg1%Arg2).str());
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
void TypesFunctions::ifBinsEdges(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  TypesFunctions::ifPoints<Arg1>(fargs);
  TypesFunctions::ifPoints<Arg2>(fargs);
  TypesFunctions::if1d<Arg1>(fargs);
  TypesFunctions::if1d<Arg2>(fargs);
  if (args[Arg1].shape[0] != (args[Arg2].shape[0]-1u)) {
    auto fmt = format("Transformation %1%: inputs %2% and %3% should sizes N and N+1, got %4% and %5%");
    throw args.error(args[Arg2], (fmt%args.name()%Arg1%Arg2%args[Arg1].shape[0]%(args[Arg2].shape[0])).str());
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
inline void TypesFunctions::binsToEdges(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (Arg >= args.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Arg index (%2% out of %3%)");
    throw args.error( DataType::undefined(),  (fmt%args.name()%Arg%args.size()).str() );
  }
  if (Ret >= rets.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Ret index (%2% out of %3%)");
    throw rets.error( DataType::undefined(), (fmt%rets.name()%Ret%rets.size()).str() );
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
inline void TypesFunctions::edgesToBins(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (Arg >= args.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Arg index (%2% out of %3%)");
    throw args.error( DataType::undefined(),  (fmt%args.name()%Arg%args.size()).str() );
  }
  if (Ret >= rets.size()) {
    auto fmt = boost::format("Transformation %1%: invalid Ret index (%2% out of %3%)");
    throw rets.error( DataType::undefined(),  (fmt%rets.name()%Ret%rets.size()).str() );
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
inline void TypesFunctions::ifHist(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
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
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not an array.
 */
template <size_t Arg>
inline void TypesFunctions::ifPoints(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
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
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not N-dimensional.
 */
template <size_t Arg, size_t Ndim>
inline void TypesFunctions::ifNd(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
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
 * @param fargs.args -- source types.
 * @param fargs.rets -- output types.
 *
 * @exception SourceTypeError in case input data is not 1-dimensional.
 */
template <size_t Arg>
inline void TypesFunctions::if1d(TransformationTypes::TypesFunctionArgs& fargs) {
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
inline void TypesFunctions::ifEmpty(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto size=args[Arg].size();
  if (size!=0u) {
    auto fmt = boost::format("Transformation %1%: Arg %2% should has 0 elements");
    throw args.error(args[Arg], (fmt%args.name()%Arg).str());
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
inline void TypesFunctions::if2d(TransformationTypes::TypesFunctionArgs& fargs) {
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
inline void TypesFunctions::ifSquare(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto shape = args[Arg].shape;
  if (shape.size()!=2 || shape[0]!=shape[1] ) {
    auto fmt = boost::format("Transformation %1%: Arg %2% should be NxN, got %3%x%4%");
    throw args.error(args[Arg], (fmt%args.name()%Arg%shape[0]%shape[1]).str());
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
inline void TypesFunctions::passAllInRange(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  size_t start=Arg1<0 ? args.size()+Arg1   : Arg1;
  size_t end  =Arg2<0 ? args.size()+Arg2+1 : Arg2+1;
  size_t ret  =Ret1<0 ? rets.size()+Ret1   : Ret1;
  for (size_t i = start; i < end; ++i) {
    if(ret>=rets.size()){
      auto fmt = format("Transformation %1%: ret %2% is out of limits (%3%)");
      throw std::runtime_error((fmt % rets.name() % ret % rets.size()).str());
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
template <int Arg1, int Ret1, int Ret2>
inline void TypesFunctions::passToRange(TransformationTypes::TypesFunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  size_t arg  =Arg1<0 ? args.size()+Arg1   : Arg1;
  size_t start=Ret1<0 ? rets.size()+Ret1   : Ret1;
  size_t end  =Ret2<0 ? rets.size()+Ret2+1 : Ret2+1;
  for (size_t i = start; i < end; ++i) {
    if(i>=rets.size()){
      auto fmt = format("Transformation %1%: ret %2% is out of limits (%3%)");
      throw std::runtime_error((fmt % rets.name() % i % rets.size()).str());
    }
    rets[i] = args[arg];
  }
}
