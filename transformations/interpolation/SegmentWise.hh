#pragma once

#include "GNAObject.hh"

class SegmentWise: public GNAObject,
                   public TransformationBind<SegmentWise> {
public:
  SegmentWise();

  void setTolerance(double value) { m_tolerance = value; }

  void determineSegments(Args, Rets);

protected:
  double m_tolerance{1.e-16};
};
