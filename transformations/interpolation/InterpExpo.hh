#pragma once

#include <string>
#include "SegmentWise.hh"

class InterpExpo: public SegmentWise,
                  public TransformationBind<InterpExpo> {
public:
  enum Strategy {
    Constant = 0,
    Extrapolate
  };
  using TransformationBind<InterpExpo>::transformation_;

  InterpExpo(const std::string& underflow_strategy="", const std::string& overflow_strategy="");

  void do_interpolate(Args, Rets);
  void interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx)a;

  void setUnderflow(double value) { m_underflow = value; }
  void setOverflow(double value) { m_overflow = value; }

  void setUnderflowStrategy(const std::string& strategy)  { m_underflow_strategy=getStrategy(strategy); }
  void setOverflowStrategy(const std::string& strategy)   { m_overflow_strategy=getStrategy(strategy); }

  void setUnderflowStrategy(Strategy strategy)  { m_underflow_strategy=strategy; }
  void setOverflowStrategy(Strategy strategy)   { m_overflow_strategy=strategy; }

  Strategy getStrategy(const std::string& strategy);
protected:
  double m_underflow{0.0};
  double m_overflow{0.0};

  Strategy m_underflow_strategy{Constant};
  Strategy m_overflow_strategy{Constant};
};
