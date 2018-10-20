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
  init();
}

IntegratorGL::IntegratorGL(size_t bins, int* orders, double* edges) : IntegratorBase(bins, orders, edges)
{
  init();
}

void IntegratorGL::init() {
  auto trans=transformation_("points")
      .output("x")
      .output("xedges")
      .types(&IntegratorGL::check)
      .func(&IntegratorGL::compute_gl)
      ;

  if(m_edges.size()){
    trans.finalize();
  }
  else{
    trans.input("edges") //hist with edges
      .types(TypesFunctions::if1d<0>, TypesFunctions::ifHist<0>, TypesFunctions::binsToEdges<0,1>);
  }
}

void IntegratorGL::check(TypesFunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[0] = DataType().points().shape(m_weights.size());

  auto& args=fargs.args;
  if(args.size() && !m_edges.size()){
    auto& edges=fargs.args[0].edges;
    m_edges=Map<const ArrayXd>(edges.data(), edges.size());
  }
  rets[1]=DataType().points().shape(m_edges.size()).preallocated(m_edges.data());
}

void IntegratorGL::compute_gl(FunctionArgs& fargs){
  auto* edge_a=m_edges.data();
  auto* edge_b{next(edge_a)};
  auto& rets=fargs.rets;
  auto *abscissa(rets[0].buffer), *weight(m_weights.data());
  rets[1].x = m_edges.cast<double>();

  GSLSamplerGL sampler;
  for (size_t i = 0; i < m_orders.size(); ++i) {
    size_t n = static_cast<size_t>(m_orders[i]);
    sampler.fill(n, *edge_a, *edge_b, abscissa, weight);
    advance(abscissa, n);
    advance(weight, n);
    advance(edge_a, 1);
    advance(edge_b, 1);
  }
}

