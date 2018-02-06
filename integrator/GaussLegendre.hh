#ifndef GAUSSLEGENDRE_H
#define GAUSSLEGENDRE_H

#include <vector>

#include <Eigen/Dense>

#include "GNAObject.hh"

class GaussLegendre: public GNAObject,
                     public Transformation<GaussLegendre> {
  friend class GaussLegendreHist;
public:
  GaussLegendre(const std::vector<double> &edges,
                const std::vector<int> &orders)
  : m_edges(edges), m_orders(orders)
  {
    init();
  }

  GaussLegendre(const std::vector<double> &edges,
                size_t orders)
  : m_edges(edges), m_orders(edges.size()-1, static_cast<int>(orders))
  {
    init();
  }

  GaussLegendre(const double *edges, const size_t *orders, size_t cnt)
    : m_edges(cnt+1), m_orders(cnt)
  {
    std::copy(edges, edges+cnt+1, m_edges.begin());
    std::copy(orders, orders+cnt, m_orders.begin());
    init();
  }

  GaussLegendre(const double *edges, size_t order, size_t cnt)
    : m_edges(cnt+1), m_orders(cnt, order)
  {
    std::copy(edges, edges+cnt+1, m_edges.begin());
    init();
  }
protected:
  void init();

  std::vector<double> m_edges;
  std::vector<int> m_orders;
  Eigen::ArrayXd m_points;
  Eigen::ArrayXd m_weights;
};

class GaussLegendreHist: public GNASingleObject,
                         public Transformation<GaussLegendreHist> {
public:
  GaussLegendreHist(const GaussLegendre *base);
protected:

  const GaussLegendre *m_base;
};

#endif // GAUSSLEGENDRE_H

