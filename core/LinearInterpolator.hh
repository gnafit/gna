#ifndef LINEARINTERPOLATOR_H
#define LINEARINTERPOLATOR_H

#include <vector>
#include <limits>

#include "GNAObject.hh"

class LinearInterpolator: public GNAObject,
                          public Transformation<LinearInterpolator> {
public:
  TransformationDef(LinearInterpolator)
  LinearInterpolator(int size, const double *xs, const double *ys)
    : m_xs(xs, xs+size), m_ys(ys, ys+size)
  {
    indexBins();
    transformation_("f")
      .input("x", DataType().points().any())
      .output("y", DataType().points().any())
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

  ClassDef(LinearInterpolator, 1);
};

#endif // LINEARINTERPOLATOR_H
