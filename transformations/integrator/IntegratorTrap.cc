#include "IntegratorTrap.hh"

#include <Eigen/Dense>
#include <iterator>

#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

IntegratorTrap::IntegratorTrap(int orders) : IntegratorBase(orders)
{
  init_sampler();
}

IntegratorTrap::IntegratorTrap(size_t bins, int orders, double* edges) : IntegratorBase(bins, orders, edges)
{
  init_sampler();
}

IntegratorTrap::IntegratorTrap(size_t bins, int* orders, double* edges) : IntegratorBase(bins, orders, edges)
{
  init_sampler();
}

void IntegratorTrap::sample(FunctionArgs& fargs) {
  auto& rets=fargs.rets;
  rets[1].x = m_edges.cast<double>();
  rets[2].x = 0.0;
  auto npoints=m_edges.size()-1;
  rets[3].x = 0.5*(m_edges.tail(npoints)+m_edges.head(npoints));

  auto& abscissa=rets[0].x;

  auto nbins=m_edges.size()-1;
  auto& binwidths=m_edges.tail(nbins) - m_edges.head(nbins);
  ArrayXd samplewidths=binwidths/(m_orders.cast<double>()-1.0);

  auto* edge_a=m_edges.data();
  auto* edge_b{next(edge_a)};

  size_t offset=0;
  for (size_t i = 0; i < static_cast<size_t>(m_orders.size()); ++i) {
    auto n=m_orders[i];
    if(n){
      abscissa.segment(offset, n)=ArrayXd::LinSpaced(n, *edge_a, *edge_b);

      auto swidth=samplewidths[i];

      auto weights=m_weights.segment(offset, n);
      weights[0]=weights[n-1]=swidth*0.5;
      if(n>2) {
        weights.segment(1, n-2)=swidth;
      }
      offset+=n;
    }
    advance(edge_a, 1);
    advance(edge_b, 1);
  }
  rets.untaint();
  rets.freeze();
}
