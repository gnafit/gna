#include <gsl/gsl_integration.h>

#include <algorithm>

#include <Eigen/Dense>

#include "GaussLegendre.hh"

using namespace Eigen;

void GaussLegendre::init() {
  size_t npoints = 0;
  for (size_t n: m_orders) {
    npoints += n;
  }
  m_points.resize(npoints);
  m_weights.resize(npoints);
  size_t k = 0;
  for (size_t i = 0; i < m_orders.size(); ++i) {
    size_t n = m_orders[i];
    auto *t = gsl_integration_glfixed_table_alloc(n);

    double a = m_edges[i];
    double b = m_edges[i+1];

    for (size_t j = 0; j < t->n; ++j) {
      gsl_integration_glfixed_point(a, b, j, &m_points[k], &m_weights[k], t);
      k++;
    }
    gsl_integration_glfixed_table_free(t);
  }
  transformation_(this, "points")
    .output("x")
    .types(&GaussLegendre::pointsTypes)
    .func(&GaussLegendre::points);
  transformation_(this, "hist")
    .types(&GaussLegendre::histTypes)
    .func(&GaussLegendre::hist);
}

void GaussLegendre::pointsTypes(Atypes /*args*/, Rtypes rets) {
  rets[0] = DataType().points().shape(m_points.size());
}

void GaussLegendre::points(Args /*args*/, Rets rets) {
  rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&m_points[0], m_points.size());
}

void GaussLegendre::addfunction(SingleOutput &out) {
  t_["hist"].input(out);
  t_["hist"].output(out);
}

void GaussLegendre::histTypes(Atypes /*args*/, Rtypes rets) {
  for (size_t k = 0; k < rets.size(); ++k) {
    rets[k] = DataType().hist().bins(m_orders.size()).edges(m_edges);
  }
}

void GaussLegendre::hist(Args args, Rets rets) {
  for (size_t k = 0; k < rets.size(); ++k) {
    ArrayXd prod = args[k].x*m_weights;
    auto *data = prod.data();
    for (size_t i = 0; i < m_orders.size(); ++i) {
      size_t n = m_orders[i];
      rets[k].x(i) = std::accumulate(data, data+n, 0.0);
      data += n;
    }
  }
}
