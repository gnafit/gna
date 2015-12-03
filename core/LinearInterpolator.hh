#ifndef LINEARINTERPOLATOR_H
#define LINEARINTERPOLATOR_H

#include <vector>
#include <limits>

#include "GNAObject.hh"

class LinearInterpolator: public GNASingleObject,
                          public Transformation<LinearInterpolator> {
public:
  LinearInterpolator(int size, const double *xs, const double *ys)
    : m_xs(xs, xs+size), m_ys(ys, ys+size)
  {
    indexBins();
    transformation_(this, "f")
      .input("x")
      .output("y")
      .types(Atypes::pass<0>)
      .func(&LinearInterpolator::interpolate)
    ;
  }
protected:
  void indexBins();
  void interpolate(Args args, Rets rets);

  std::vector<double> m_xs;
  std::vector<double> m_ys;
  std::vector<size_t> m_index;
  double m_minbinsize;
};

#endif // LINEARINTERPOLATOR_H
