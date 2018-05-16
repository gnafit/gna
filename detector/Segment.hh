#ifndef SEGMENT_H
#define SEGMENT_H

#include <vector>

#include "GNAObject.hh"
#include "Eigen/Sparse"

class Segment: public GNASingleObject,
                        public Transformation<Segment> {
public:
  Segment();

  std::vector<std::double> calweights() const noexcept;

private:

  variable<double> m_a, m_b, m_c;


};

#endif // SEGMENT_H
