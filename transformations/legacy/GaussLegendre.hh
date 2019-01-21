#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "GNAObject.hh"

class GaussLegendre: public GNAObject,
                     public TransformationBind<GaussLegendre> {
  friend class GaussLegendreHist;
public:
  GaussLegendre(std::vector<double> edges,
                std::vector<int> orders)
  : m_edges(std::move(edges)), m_orders(std::move(orders))
  {
    init();
  }

  GaussLegendre(const std::vector<double> &edges,
                size_t orders)
  : m_edges(edges), m_orders(edges.size()-1, static_cast<int>(orders))
  {
    init();
  }

  GaussLegendre(const double *edges, const size_t *orders, size_t bins)
    : m_edges(bins+1), m_orders(bins)
  {
    std::copy(edges, edges+bins+1, m_edges.begin());
    std::copy(orders, orders+bins, m_orders.begin());
    init();
  }

  GaussLegendre(const double *edges, size_t order, size_t bins)
    : m_edges(bins+1), m_orders(bins, order)
  {
    std::copy(edges, edges+bins+1, m_edges.begin());
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
                         public TransformationBind<GaussLegendreHist> {
public:
  GaussLegendreHist(const GaussLegendre *base);
protected:

  const GaussLegendre *m_base;
};
