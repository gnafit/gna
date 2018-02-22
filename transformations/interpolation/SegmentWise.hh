#pragma once

#include "GNAObject.hh"

class SegmentWise: public GNASingleObject,
                   public TransformationBind<SegmentWise> {
public:
  SegmentWise();

  void setTolerance(double value) { m_tolerance = value; }
protected:
  void determineSegments(Args, Rets);

  double m_tolerance{1.e-16};
};
