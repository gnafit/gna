#pragma once

#include "GNAObject.hh"

class SegmentWise: public GNASingleObject,
                   public TransformationBind<SegmentWise> {
public:
  SegmentWise(size_t nedges, const double* edges);

protected:
  void defineTypes(Atypes, Rtypes);
  void determineSegments(Args, Rets);

  Eigen::ArrayXd m_edges;  ///< The array holding the edges.
};
