#ifndef GAUSSLEGENDRE2D_H
#define GAUSSLEGENDRE2D_H

#include <vector>

#include <Eigen/Dense>

#include "GNAObject.hh"

class GaussLegendre2d: public GNAObject,
                       public Transformation<GaussLegendre2d> {
public:
  TransformationDef(GaussLegendre2d)
  GaussLegendre2d(const std::vector<double> &xedges,
                  const std::vector<int> &xorders,
                  double ymin, double ymax, int yorder)
  : m_xedges(xedges), m_xorders(xorders),
    m_ymin(ymin), m_ymax(ymax), m_yorder(yorder)
  {
    init();
  }
  GaussLegendre2d(const double *xedges, const size_t *xorders, size_t xcnt,
                double ymin, double ymax, int yorder)
    : m_xedges(xcnt), m_xorders(xcnt),
      m_ymin(ymin), m_ymax(ymax), m_yorder(yorder)
  {
    std::copy(xedges, xedges+xcnt+1, m_xedges.begin());
    std::copy(xorders, xorders+xcnt, m_xorders.begin());
    init();
  }
protected:
  void init();

  Status pointsTypes(Atypes args, Rtypes rets);
  Status points(Args args, Rets rets);

  Status histTypes(Atypes args, Rtypes rets);
  Status hist(Args args, Rets rets);

  std::vector<double> m_xedges;
  std::vector<int> m_xorders;
  double m_ymin, m_ymax;
  int m_yorder;

  std::vector<double> m_xpoints;
  Eigen::ArrayXd m_xweights;

  std::vector<double> m_ypoints;
  Eigen::ArrayXd m_yweights;

  ClassDef(GaussLegendre2d, 1);
};

#endif // GAUSSLEGENDRE2D_H

