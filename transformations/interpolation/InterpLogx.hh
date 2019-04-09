#pragma once

#include <string>
#include "InSegment.hh"

class InterpLogx: public InSegment,
                  public TransformationBind<InterpLogx> {
public:
  //enum Strategy { ///< Extrapolation strategy.
    //Constant = 0, ///< Fill with constant value.
    //Extrapolate   ///< Extrapolate using first/last segment function.
  //};
  using TransformationBind<InterpLogx>::transformation_;

  //InterpLogx(const std::string& underflow_strategy="", const std::string& overflow_strategy="");             ///< Constructor.
  InterpLogx();                                                                                                ///< Constructor.
  InterpLogx(SingleOutput& x, SingleOutput& newx);                                                             ///< Constructor.
  InterpLogx(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                                            ///< Constructor.

  TransformationDescriptor add_transformation(const std::string& name="");
  void bind_transformations();
  void bind_inputs();
  void set(SingleOutput& x, SingleOutput& newx);

  OutputDescriptor interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                        ///< Initialize transformations by connecting `x`, `y` and `newy` outputs.

  //void setUnderflow(double value) { m_underflow = value; }                                                 ///< Set value to write into underflow points when strategy=Constant.
  //void setOverflow(double value) { m_overflow = value; }                                                   ///< Set value to write into overflow points when strategy=Constant.

  //void setUnderflowStrategy(const std::string& strategy)  { m_underflow_strategy=getStrategy(strategy); }  ///< Set strategy to use for underflow points: 'constant' or 'extrapolate'.
  //void setOverflowStrategy(const std::string& strategy)   { m_overflow_strategy=getStrategy(strategy); }   ///< Set strategy to use for overflow points: 'constant' or 'extrapolate'.

  //void setUnderflowStrategy(Strategy strategy)  { m_underflow_strategy=strategy; }                         ///< Set strategy to use for underflow points: `InterpLogx::Constant` or `InterpLogx::Extrapolate`.
  //void setOverflowStrategy(Strategy strategy)   { m_overflow_strategy=strategy; }                          ///< Set strategy to use for overflow points:  `InterpLogx::Constant` or `InterpLogx::Extrapolate`.

  //Strategy getStrategy(const std::string& strategy);                                                       ///< Convert strategy from string to Strategy.

  void do_interpolate(FunctionArgs& fargs);                                                                    ///< Do the interpolation.
protected:
  //double m_underflow{0.0};                                                                                 ///< Value to write into underflow points when strategy=Constant.
  //double m_overflow{0.0};                                                                                  ///< Value to write into overflow points when strategy=Constant.

  //Strategy m_underflow_strategy{Constant};                                                                 ///< Strategy to use for underflow points.
  //Strategy m_overflow_strategy{Constant};                                                                  ///< Strategy to use for overflow points.
};
