#ifndef GAUSSLEGENDRE_H
#define GAUSSLEGENDRE_H

#include <vector>

#include <Eigen/Dense>

#include "GNAObject.hh"

class GaussLegendre: public GNAObject,
                     public Transformation<GaussLegendre> {
public:
  TransformationDef(GaussLegendre)
  GaussLegendre(const std::vector<double> &edges,
                const std::vector<int> &orders)
  : m_edges(edges), m_orders(orders)
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
protected:
  void init();

  void pointsTypes(Atypes args, Rtypes rets);
  void points(Args args, Rets rets);

  void histTypes(Atypes args, Rtypes rets);
  void hist(Args args, Rets rets);

  std::vector<double> m_edges;
  std::vector<int> m_orders;
  std::vector<double> m_points;
  Eigen::ArrayXd m_weights;

  ClassDef(GaussLegendre, 1);
};

#endif // GAUSSLEGENDRE_H

