#pragma once

#include <vector>

#include "GNAObject.hh"
#include <Eigen/Sparse>

class Segment: public GNASingleObject,
               public TransformationBind<Segment> {
public:
  Segment();

  std::vector<std::double> calweights() const noexcept;

private:

  variable<double> m_a, m_b, m_c;


};
