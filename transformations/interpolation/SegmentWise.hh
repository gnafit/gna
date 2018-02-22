#pragma once

#include "GNAObject.hh"

class SegmentWise: public GNASingleObject,
                   public TransformationBind<SegmentWise> {
public:
  SegmentWise();

protected:
  void defineTypes(Atypes, Rtypes);
  void determineSegments(Args, Rets);
};
