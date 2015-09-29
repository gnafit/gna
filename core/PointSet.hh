#ifndef POINTSET_H
#define POINTSET_H

#include <vector>

#include "GNAObject.hh"

class PointSet: public GNAObject,
                public Transformation<PointSet> {
public:
  TransformationDef(PointSet)
  PointSet(const std::vector<double> &points)
    : m_points(points)
  {
    init();
  }
  PointSet(const double *points, size_t cnt)
    : m_points(cnt)
  {
    std::copy(points, points+cnt, m_points.begin());
    init();
  }
  const std::vector<double> &points() const { return m_points; }
protected:
  void init() {
    transformation_("points")
      .output("points", DataType().points().size(m_points.size()))
      .types([](PointSet *obj, Atypes /*args*/, Rtypes rets) {
          rets[0] = DataType().points().size(obj->points().size());
          return Status::Success;
        })
      .func([](PointSet *obj, Args /*args*/, Rets rets) {
          auto &pts = obj->points();
          rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&pts[0], pts.size());
          return Status::Success;
        });
  }
  std::vector<double> m_points;

  ClassDef(PointSet, 1);
};

#endif // POINTSET_H
