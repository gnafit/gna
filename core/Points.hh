#ifndef POINTS_H
#define POINTS_H

#include <vector>

#include "GNAObject.hh"

class Points: public GNASingleObject,
              public Transformation<Points> {
public:
  Points(const std::vector<double> &points)
    : m_points(points)
  {
    init();
  }
  Points(const double *points, size_t cnt)
    : m_points(cnt)
  {
    std::copy(points, points+cnt, m_points.begin());
    init();
  }
protected:
  void init() {
    transformation_(this, "points")
      .output("points", DataType().points().shape(m_points.size()))
      .types([](Points *obj, Atypes /*args*/, Rtypes rets) {
          rets[0] = DataType().points().shape(obj->m_points.size());
        })
      .func([](Points *obj, Args /*args*/, Rets rets) {
          auto &pts = obj->m_points;
          rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&pts[0], pts.size());
        });
  }
  std::vector<double> m_points;
};

#endif // POINTS_H
