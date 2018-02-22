#pragma once

#include "SegmentWise.hh"

class InterpExpo: public SegmentWise,
                  public TransformationBind<InterpExpo> {
public:
  using TransformationBind<InterpExpo>::transformation_;

  InterpExpo();

  void do_interpolate(Args, Rets);

  void setUnderflow(double value) { m_underflow = value; }
  void setOverflow(double value) { m_overflow = value; }
protected:
  double m_underflow{0.0};
  double m_overflow{0.0};

};
