#include "IntegratorGL.hh"

#include <Eigen/Dense>
#include <map>
#include <iterator>

#include "GSLSamplerGL.hh"
#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

IntegratorGL::IntegratorGL(size_t bins, int orders, double* edges) : IntegratorBase(bins, orders, edges)
{
  init_sampler();
}

IntegratorGL::IntegratorGL(size_t bins, int* orders, double* edges) : IntegratorBase(bins, orders, edges)
{
  init_sampler();
}

void IntegratorGL::sample(FunctionArgs& fargs){
  auto& rets=fargs.rets;
  GSLSamplerGL sampler;
  sampler.fill_bins(m_orders.size(), m_orders.data(), m_edges.data(), rets[0].buffer, m_weights.data());
  rets[1].x = m_edges.cast<double>();
}

