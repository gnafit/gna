#include "IntegratorTrap.hh"

#include <Eigen/Dense>
#include <iterator>

#include "TypesFunctions.hh"

using namespace Eigen;
using namespace std;

IntegratorTrap::IntegratorTrap(size_t bins, int orders, double* edges) : IntegratorBase(bins, orders, edges)
{
  init();
}

IntegratorTrap::IntegratorTrap(size_t bins, int* orders, double* edges) : IntegratorBase(bins, orders, edges)
{
  init();
}

void IntegratorTrap::init() {
  set_shared_edge();
  auto trans=transformation_("points")
      .output("x")
      .output("xedges")
      .types(&IntegratorTrap::check)
      .func(&IntegratorTrap::compute_trap)
      ;

  if(m_edges.size()){
    trans.finalize();
  }
  else{
    trans.input("edges") //hist with edges
      .types(TypesFunctions::if1d<0>, TypesFunctions::ifHist<0>, TypesFunctions::binsToEdges<0,1>);
  }
}

void IntegratorTrap::check(TypesFunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[0] = DataType().points().shape(m_weights.size());

  auto& args=fargs.args;
  if(args.size() && !m_edges.size()){
    auto& edges=fargs.args[0].edges;
    m_edges=Map<const ArrayXd>(edges.data(), edges.size());
  }
  rets[1]=DataType().points().shape(m_edges.size()).preallocated(m_edges.data());
}

void IntegratorTrap::compute_trap(FunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[1].x = m_edges.cast<double>();
  auto& abscissa=rets[0].x;

  auto nbins=m_edges.size()-1;
  auto& binwidths=m_edges.tail(nbins) - m_edges.head(nbins);
  ArrayXd samplewidths=binwidths/m_orders.cast<double>();

  auto* edge_a=m_edges.data();
  auto* edge_b{next(edge_a)};

  size_t offset=0;
  for (size_t i = 0; i < m_orders.size(); ++i) {
    auto n=m_orders[i]+1;
    abscissa.segment(offset, n)=ArrayXd::LinSpaced(n, *edge_a, *edge_b);

    auto swidth=samplewidths[i];
    m_weights[offset]=swidth*0.5;
    if(n>2) {
      m_weights.segment(offset+1, n-2)=swidth;
    }

    offset+=n-1;
    advance(edge_a, 1);
    advance(edge_b, 1);
  }
  m_weights.tail(1)=samplewidths.tail(1)*0.5;
}
