#include "IntegratorBase.hh"
#include "fmt/format.h"

#include <string>
#include <iostream>
#include <iterator>

using namespace Eigen;
using namespace std;

IntegratorBase::IntegratorBase(size_t bins, int orders, double* edges, bool shared_edge) :
m_orders(bins),
m_shared_edge(static_cast<size_t>(shared_edge))
{
    m_orders.setConstant(orders);
    init_base(edges);
}

IntegratorBase::IntegratorBase(size_t bins, int *orders, double* edges, bool shared_edge) :
m_orders(Map<const ArrayXi>(orders, bins)),
m_shared_edge(static_cast<size_t>(shared_edge))
{
    init_base(edges);
}

void IntegratorBase::init_base(double* edges) {
    if(m_shared_edge){
      m_weights.resize(m_orders.sum()-m_orders.size()+m_shared_edge);
    }
    else{
      m_weights.resize(m_orders.sum());
    }
    if(edges){
        m_edges=Map<const ArrayXd>(edges, m_orders.size()+1);
    }
    add();
}

TransformationDescriptor IntegratorBase::add(){
    int num=transformations.size()-1;
    std::string name="hist";
    if(num>0){
      name = fmt::format("{0}_{1:02d}", name, num);
    }
    transformation_(name)
        .input("f")
        .output("hist")
        .types(TypesFunctions::if1d<0>, TypesFunctions::ifPoints<0>, &IntegratorBase::check_base)
        .func(&IntegratorBase::integrate)
        ;
    return transformations.back();
}

void IntegratorBase::check_base(TypesFunctionArgs& fargs){
    fargs.rets[0]=DataType().hist().edges(m_edges.size(), m_edges.data());

    if (fargs.args[0].shape[0] != static_cast<size_t>(m_weights.size())){
        throw fargs.args.error(fargs.args[0], "inconsistent function size");
    }

    if((m_orders<(1+m_shared_edge)).any()){
        std::cerr<<m_orders<<std::endl;
        if(m_shared_edge){
          throw std::runtime_error("All integration orders should be >=2");
        }
        else{
          throw std::runtime_error("All integration orders should be >=1");
        }
    }
}

void IntegratorBase::integrate(FunctionArgs& fargs){
    auto* ret=fargs.rets[0].buffer;

    ArrayXd prod = fargs.args[0].x*m_weights;
    auto* data = prod.data();
    auto* order = m_orders.data();
    for (int i = 0; i < m_orders.size(); ++i) {
        auto* data_next=std::next(data, *order);
        *ret = std::accumulate(data, data_next, 0.0);
        if(m_shared_edge){
            data=prev(data_next);
        }
        else{
            data=data_next;
        }
        advance(order,1);
        advance(ret,1);
    }
}

void IntegratorBase::init_sampler() {
  auto trans=transformation_("points")
      .output("x")
      .output("xedges")
      .types(&IntegratorBase::check_sampler)
      .func(&IntegratorBase::sample)
      ;

  if(m_edges.size()){
    trans.finalize();
  }
  else{
    trans.input("edges") //hist with edges
      .types(TypesFunctions::if1d<0>, TypesFunctions::ifHist<0>, TypesFunctions::binsToEdges<0,1>);
  }
}

void IntegratorBase::check_sampler(TypesFunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[0] = DataType().points().shape(m_weights.size());

  auto& args=fargs.args;
  if(args.size() && !m_edges.size()){
    auto& edges=fargs.args[0].edges;
    m_edges=Map<const ArrayXd>(edges.data(), edges.size());
  }
  rets[1]=DataType().points().shape(m_edges.size()).preallocated(m_edges.data());
}
void IntegratorBase::dump(){
    std::cout<<"Edges: "<<m_edges.transpose()<<std::endl;
    std::cout<<"Orders: "<<m_orders.transpose()<<std::endl;
    std::cout<<"Weights: "<<m_weights.transpose()<<std::endl;
}
