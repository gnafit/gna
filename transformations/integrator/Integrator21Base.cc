#include "Integrator21Base.hh"
#include "fmt/format.h"

#include <iostream>
#include <iterator>

using namespace Eigen;
using namespace std;

Integrator21Base::Integrator21Base(size_t xbins, int xorders, double* xedges, int yorder, double ymin, double ymax) :
m_xorders(xbins),
m_yorder(yorder),
m_ymin(ymin),
m_ymax(ymax),
m_yweights(yorder)
{
    m_xorders.setConstant(xorders);
    init_base(xedges);
}

Integrator21Base::Integrator21Base(size_t xbins, int* xorders, double* xedges, int yorder, double ymin, double ymax) :
m_xorders(Map<const ArrayXi>(xorders, xbins)),
m_yorder(yorder),
m_ymin(ymin),
m_ymax(ymax),
m_yweights(yorder)
{
    init_base(xedges);
}

void Integrator21Base::init_base(double* xedges) {
    m_xweights.resize(m_xorders.sum());
    if(xedges){
        m_xedges=Map<const ArrayXd>(xedges, m_xorders.size()+1);
    }

    add();
}

TransformationDescriptor Integrator21Base::add(){
    int num=transformations.size()-1;
    std::string name="hist";
    if(num>0){
      name = fmt::format("{0}_{1:02d}", name, num);
    }
    transformation_(name)
        .input("f")
        .output("hist")
        .types(TypesFunctions::ifPoints<0>, &Integrator21Base::check_base)
        .func(&Integrator21Base::integrate)
        ;
    return transformations.back();
}

void Integrator21Base::check_base(TypesFunctionArgs& fargs){
    fargs.rets[0]=DataType().hist().edges(m_xedges.size(), m_xedges.data());

    auto& shape=fargs.args[0].shape;
    if (shape[0]!=static_cast<size_t>(m_xweights.size()) || shape[1]!=static_cast<size_t>(m_yweights.size())){
        throw fargs.args.error(fargs.args[0],
                               fmt::format("Inconsistent function size {:d}x{:d}, "
                                           "should be {:d}x{:d}",
                                           shape[0], shape[1], m_xweights.size(), m_yweights.size()
                                           )
                               );
    }

    if((m_xorders<1).any() || m_yorder<1){
        std::cerr<<"X orders: "<<m_xorders<<std::endl;
        std::cerr<<"Y order: "<<m_yorder<<std::endl;
        throw std::runtime_error("All integration orders should be >=1");
    }
}

void Integrator21Base::integrate(FunctionArgs& fargs){
    auto& arg=fargs.args[0];
    auto& ret=fargs.rets[0];

    ArrayXd prod = (arg.arr2d*m_weights).rowwise().sum();
    auto* data_start = prod.data();
    for (size_t i = 0; i < static_cast<size_t>(m_xorders.size()); ++i) {
        size_t n = m_xorders[i];
        auto* data_end=std::next(data_start, n);
        ret.x(i) = std::accumulate(data_start, data_end, 0.0);
        data_start=data_end;
    }
}

void Integrator21Base::init_sampler() {
  auto trans=transformation_("points")
      .output("x")
      .output("y")
      .output("xedges")
      .output("xmesh")
      .output("ymesh")
      .types(&Integrator21Base::check_sampler)
      .func(&Integrator21Base::sample)
      ;

  if(m_xedges.size()){
    trans.finalize();
  }
  else{
    trans.input("edges") //hist with edges
      .types(TypesFunctions::if1d<0>, TypesFunctions::ifHist<0>, TypesFunctions::binsToEdges<0,2>);
  }
}

void Integrator21Base::check_sampler(TypesFunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[0] = DataType().points().shape(m_xweights.size());
  rets[1] = DataType().points().shape(m_yweights.size());

  rets[3] = DataType().points().shape(m_xweights.size(), m_yweights.size());
  rets[4] = DataType().points().shape(m_xweights.size(), m_yweights.size());

  auto& args=fargs.args;
  if(args.size() && !m_xedges.size()){
    auto& edges=fargs.args[0].edges;
    m_xedges=Map<const ArrayXd>(edges.data(), edges.size());
  }
  rets[2]=DataType().points().shape(m_xedges.size()).preallocated(m_xedges.data());
}

void Integrator21Base::dump(){
    std::cout<<"X edges: "<<m_xedges.transpose()<<std::endl;
    std::cout<<"X orders: "<<m_xorders.transpose()<<std::endl;
    std::cout<<"X weights: "<<m_xweights.transpose()<<std::endl;
    std::cout<<"Y edges: "<<m_ymin<<" "<<m_ymax<<std::endl;
    std::cout<<"Y order: "<<m_yorder<<std::endl;
    std::cout<<"Y weights: "<<m_yweights.transpose()<<std::endl;

}
