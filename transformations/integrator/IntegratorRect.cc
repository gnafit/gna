#include "IntegratorRect.hh"

#include <Eigen/Dense>
#include <iterator>

#include "SamplerRect.hh"
#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

IntegratorRect::IntegratorRect(int order, const std::string& mode) :
IntegratorBase(order),
m_mode(mode)
{
  m_rect_offset = SamplerRect::offset(mode);
  init_sampler();
}

IntegratorRect::IntegratorRect(size_t bins, int orders, double* edges, const std::string& mode) :
IntegratorBase(bins, orders, edges),
m_mode(mode)
{
  m_rect_offset = SamplerRect::offset(mode);
  init_sampler();
}

IntegratorRect::IntegratorRect(size_t bins, int* orders, double* edges, const std::string& mode) :
IntegratorBase(bins, orders, edges),
m_mode(mode)
{
  m_rect_offset = SamplerRect::offset(mode);
  init_sampler();
}

void IntegratorRect::sample(FunctionArgs& fargs){
  auto& rets=fargs.rets;
  SamplerRect::fill_bins(m_rect_offset, m_orders.size(), m_orders.data(), m_edges.data(), rets[0].buffer, m_weights.data());
  rets[1].x = m_edges.cast<double>();
  rets[2].x = 0.0;
  auto npoints=m_edges.size()-1;
  rets[3].x = 0.5*(m_edges.tail(npoints)+m_edges.head(npoints));
  rets.untaint();
  rets.freeze();
}
