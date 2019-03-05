#ifndef POINTS_H
#define POINTS_H

#include <vector>

#include "GNAObject.hh"

class Points: public GNASingleObject,
              public Transformation<Points> {
public:
  Points(const std::vector<double> &points)
    : m_points(points), m_shape{points.size()}
  {
    init();
  }
  Points(const double *points, size_t cnt)
    : m_points(cnt), m_shape{cnt}
  {
    std::copy(points, points+cnt, m_points.begin());
    init();
  }
  Points(const double *points, std::vector<size_t> shape)
    : m_shape(shape)
  {
    size_t cnt = 1;
    for (auto x: shape) {
      cnt *= x;
    }
    m_points.resize(cnt);
    std::copy(points, points+cnt, m_points.begin());
    init();
  }

  size_t size() const {
    return m_points.size();
  }
  const double *data() const {
    return &m_points[0];
  }
protected:
  void init() {
    transformation_(this, "points")
      .output("points")
      .types([](Points *obj, Atypes /*args*/, Rtypes rets) {
          rets[0] = DataType().points().shape(obj->m_shape);
        })
      .func([](Points *obj, Args /*args*/, Rets rets) {
          auto &pts = obj->m_points;
          rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&pts[0], pts.size());
        })
      .finalize();
  }
  std::vector<double> m_points;
  std::vector<size_t> m_shape;
};

#endif // POINTS_H
