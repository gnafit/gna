#pragma once

#include <string>
#include "InSegment.hh"

/**
 * @brief Linear interpolation (unordered).
 *
 * For a given `x`, `y` and `newx` computes `newy` via linear approximation:
 *   - f(x) = k_i * (x-x_i) + y_i
 *
 * The object inherits InSegment that determines the segments to be used for interpolation.
 *
 * @note No check is performed on the value of `b`.
 *
 * Inputs:
 *   - interp.newx -- array with points with fine steps to interpolate on.
 *   - interp.x -- array with coarse points.
 *   - interp.y -- values of a function on `x`.
 *   - interp.insegment -- segments of `newx` in `x`. See InSegment.
 *
 * Outputs:
 *   - interp.interp
 *
 * The connection may be done via InterpLinear::interpolate() method.
 *
 * @author Maxim Gonchar
 * @date 02.2017
 */
class InterpLinear: public InSegment,
                  public TransformationBind<InterpLinear> {
public:
  //enum Strategy { ///< Extrapolation strategy.
    //Constant = 0, ///< Fill with constant value.
    //Extrapolate   ///< Extrapolate using first/last segment function.
  //};
  using TransformationBind<InterpLinear>::transformation_;

  //InterpLinear(const std::string& underflow_strategy="", const std::string& overflow_strategy="");             ///< Constructor.
  InterpLinear();                                                                                                ///< Constructor.
  InterpLinear(SingleOutput& x, SingleOutput& newx);                                                             ///< Constructor.
  InterpLinear(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                                            ///< Constructor.

  TransformationDescriptor add_transformation(bool bind=true);
  void bind_transformations();
  void set(SingleOutput& x, SingleOutput& newx);
  InputDescriptor  add_input();
  OutputDescriptor add_input(SingleOutput& y);

  OutputDescriptor interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                        ///< Initialize transformations by connecting `x`, `y` and `newy` outputs.

  //void setUnderflow(double value) { m_underflow = value; }                                                 ///< Set value to write into underflow points when strategy=Constant.
  //void setOverflow(double value) { m_overflow = value; }                                                   ///< Set value to write into overflow points when strategy=Constant.

  //void setUnderflowStrategy(const std::string& strategy)  { m_underflow_strategy=getStrategy(strategy); }  ///< Set strategy to use for underflow points: 'constant' or 'extrapolate'.
  //void setOverflowStrategy(const std::string& strategy)   { m_overflow_strategy=getStrategy(strategy); }   ///< Set strategy to use for overflow points: 'constant' or 'extrapolate'.

  //void setUnderflowStrategy(Strategy strategy)  { m_underflow_strategy=strategy; }                         ///< Set strategy to use for underflow points: `InterpLinear::Constant` or `InterpLinear::Extrapolate`.
  //void setOverflowStrategy(Strategy strategy)   { m_overflow_strategy=strategy; }                          ///< Set strategy to use for overflow points:  `InterpLinear::Constant` or `InterpLinear::Extrapolate`.

  //Strategy getStrategy(const std::string& strategy);                                                       ///< Convert strategy from string to Strategy.

  void do_interpolate(FunctionArgs& fargs);                                                                    ///< Do the interpolation.
protected:
  //double m_underflow{0.0};                                                                                 ///< Value to write into underflow points when strategy=Constant.
  //double m_overflow{0.0};                                                                                  ///< Value to write into overflow points when strategy=Constant.

  //Strategy m_underflow_strategy{Constant};                                                                 ///< Strategy to use for underflow points.
  //Strategy m_overflow_strategy{Constant};                                                                  ///< Strategy to use for overflow points.
};
