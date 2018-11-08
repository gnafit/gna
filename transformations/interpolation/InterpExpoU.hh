#pragma once

#include <string>
#include "InSegment.hh"

/**
 * @brief Exponential interpolation (unordered).
 *
 * For a given `x`, `y` and `newx` computes `newy` via exponential:
 *   - f(x) = k_i * exp( -(x-x_i)*b_i ) for x in [x_i, x_i+1)
 *   - k = f(x_0)
 *   - b is chosen such that function was continuous:
 *   - b_i = - ln(k_i+1 / k_i) / ( x_i+1 - x_i )
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
 * The connection may be done via InterpExpoU::interpolate() method.
 *
 * @author Maxim Gonchar
 * @date 02.2017
 */
class InterpExpoU: public InSegment,
                   public TransformationBind<InterpExpoU> {
public:
  //enum Strategy { ///< Extrapolation strategy.
    //Constant = 0, ///< Fill with constant value.
    //Extrapolate   ///< Extrapolate using first/last segment function.
  //};
  using TransformationBind<InterpExpoU>::transformation_;

  //InterpExpoU(const std::string& underflow_strategy="", const std::string& overflow_strategy="");           ///< Constructor.
  InterpExpoU();                                                                                              ///< Constructor.

  void do_interpolate(FunctionArgs& fargs);                                                                   ///< Do the interpolation.
  void interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                                     ///< Initialize transformations by connecting `x`, `y` and `newy` outputs.
  void interpolate(TransformationDescriptor& insegment, SingleOutput& x, SingleOutput& y, SingleOutput& newx); ///< Initialize transformations by connecting `x`, `y` and `newy` outputs.

  //void setUnderflow(double value) { m_underflow = value; }                                                 ///< Set value to write into underflow points when strategy=Constant.
  //void setOverflow(double value) { m_overflow = value; }                                                   ///< Set value to write into overflow points when strategy=Constant.

  //void setUnderflowStrategy(const std::string& strategy)  { m_underflow_strategy=getStrategy(strategy); }  ///< Set strategy to use for underflow points: 'constant' or 'extrapolate'.
  //void setOverflowStrategy(const std::string& strategy)   { m_overflow_strategy=getStrategy(strategy); }   ///< Set strategy to use for overflow points: 'constant' or 'extrapolate'.

  //void setUnderflowStrategy(Strategy strategy)  { m_underflow_strategy=strategy; }                         ///< Set strategy to use for underflow points: `InterpExpoU::Constant` or `InterpExpoU::Extrapolate`.
  //void setOverflowStrategy(Strategy strategy)   { m_overflow_strategy=strategy; }                          ///< Set strategy to use for overflow points:  `InterpExpoU::Constant` or `InterpExpoU::Extrapolate`.

  //Strategy getStrategy(const std::string& strategy);                                                       ///< Convert strategy from string to Strategy.
protected:
  //double m_underflow{0.0};                                                                                 ///< Value to write into underflow points when strategy=Constant.
  //double m_overflow{0.0};                                                                                  ///< Value to write into overflow points when strategy=Constant.

  //Strategy m_underflow_strategy{Constant};                                                                 ///< Strategy to use for underflow points.
  //Strategy m_overflow_strategy{Constant};                                                                  ///< Strategy to use for overflow points.
};
