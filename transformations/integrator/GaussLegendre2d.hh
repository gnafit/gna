#ifndef GAUSSLEGENDRE2D_H
#define GAUSSLEGENDRE2D_H

#include <vector>

#include <Eigen/Dense>

#include "GNAObject.hh"

class GaussLegendre2d: public GNAObject,
                       public TransformationBind<GaussLegendre2d> {
  friend class GaussLegendre2dHist;
public:
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
    : m_xedges(xcnt+1), m_xorders(xcnt),
      m_ymin(ymin), m_ymax(ymax), m_yorder(yorder)
  {
    std::copy(xedges, xedges+xcnt+1, m_xedges.begin());
    std::copy(xorders, xorders+xcnt, m_xorders.begin());
    init();
  }
protected:
  void init();

  void pointsTypes(Atypes args, Rtypes rets);
  void points(Args args, Rets rets);

  void histTypes(Atypes args, Rtypes rets);
  void hist(Args args, Rets rets);

  std::vector<double> m_xedges;
  std::vector<int> m_xorders;
  double m_ymin, m_ymax;
  int m_yorder;

  std::vector<double> m_xpoints;
  Eigen::ArrayXd m_xweights;

  std::vector<double> m_ypoints;
  Eigen::ArrayXd m_yweights;
};

class GaussLegendre2dHist: public GNASingleObject,
                           public TransformationBind<GaussLegendre2dHist> {
public:
  GaussLegendre2dHist(const GaussLegendre2d *base);
protected:
  const GaussLegendre2d *m_base;
};

#endif // GAUSSLEGENDRE2D_H

