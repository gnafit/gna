#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "GNAObject.hh"

class GaussLegendre2d: public GNAObject,
                       public TransformationBind<GaussLegendre2d> {
  friend class GaussLegendre2dHist;
public:
  GaussLegendre2d(std::vector<double> xedges,
                  std::vector<int> xorders,
                  double ymin, double ymax, int yorder)
  : m_xedges(std::move(xedges)), m_xorders(std::move(xorders)),
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
  GaussLegendre2d(const double *xedges, size_t xorder, size_t xcnt,
                double ymin, double ymax, int yorder)
    : m_xedges(xcnt+1), m_xorders(xcnt, xorder),
      m_ymin(ymin), m_ymax(ymax), m_yorder(yorder)
  {
    std::copy(xedges, xedges+xcnt+1, m_xedges.begin());
    init();
  }
protected:
  void init();

  void pointsTypes(TypesFunctionArgs& fargs);
  void points(FunctionArgs& fargs);

  void histTypes(TypesFunctionArgs& fargs);
  void hist(FunctionArgs& fargs);

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

