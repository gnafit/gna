#include "Integrator2Base.hh"
#include "fmt/format.h"

#include <iostream>
#include <iterator>

using namespace Eigen;
using namespace std;

Integrator2Base::Integrator2Base(size_t xbins, int xorders, double* xedges, size_t ybins, int yorders, double* yedges) :
m_xorders(xbins),
m_yorders(ybins)
{
    m_xorders.setConstant(xorders);
    m_yorders.setConstant(yorders);
    init_base(xedges, yedges);
}

Integrator2Base::Integrator2Base(size_t xbins, int* xorders, double* xedges, size_t ybins, int* yorders, double* yedges) :
m_xorders(Map<const ArrayXi>(xorders, xbins)),
m_yorders(Map<const ArrayXi>(yorders, ybins))
{
    init_base(xedges, yedges);
}

void Integrator2Base::init_base(double* xedges, double* yedges) {
    if(bool(xedges)^bool(yedges)){
        throw std::runtime_error("xedges and yedges should be either both defined or both not defined");
    }

    m_xweights.resize(m_xorders.sum());
    if(xedges){
        m_xedges=Map<const ArrayXd>(xedges, m_xorders.size()+1);
    }

    m_yweights.resize(m_yorders.sum());
    if(yedges){
        m_yedges=Map<const ArrayXd>(yedges, m_yorders.size()+1);
    }

    add();
}

TransformationDescriptor Integrator2Base::add(){
    int num=transformations.size()-1;
    std::string name="hist";
    if(num>0){
      name = fmt::format("{0}_{1:02d}", name, num);
    }
    transformation_(name)
        .input("f")
        .output("hist")
        .types(TypesFunctions::ifPoints<0>, TypesFunctions::if2d<0>, &Integrator2Base::check_base)
        .func(&Integrator2Base::integrate)
        ;
    return transformations.back();
}

void Integrator2Base::check_base(TypesFunctionArgs& fargs){
    fargs.rets[0]=DataType().hist().edges(m_xedges.size(), m_xedges.data(), m_yedges.size(), m_yedges.data());

    auto& shape=fargs.args[0].shape;
    if (shape[0]!=static_cast<size_t>(m_xweights.size()) || shape[1]!=static_cast<size_t>(m_yweights.size())){
        throw fargs.args.error(fargs.args[0],
                               fmt::format("Inconsistent function size {:d}x{:d}, "
                                           "should be {:d}x{:d}",
                                           shape[0], shape[1], m_xweights.size(), m_yweights.size()
                                           )
                               );
    }

    if((m_xorders<1).any() || (m_yorders<1).any()){
        std::cerr<<"X orders: "<<m_xorders<<std::endl;
        std::cerr<<"Y orders: "<<m_yorders<<std::endl;
        throw std::runtime_error("All integration orders should be >=1");
    }
}

void Integrator2Base::integrate(FunctionArgs& fargs){

    auto& arg=fargs.args[0];
    auto& ret=fargs.rets[0];

    ArrayXXd prod = arg.arr2d*m_weights;
    size_t x_offset=0;
    for (size_t ix = 0; ix < static_cast<size_t>(m_xorders.size()); ++ix) {
        size_t nx = m_xorders[ix];
        size_t y_offset=0;
        for (size_t iy = 0; iy < static_cast<size_t>(m_yorders.size()); ++iy) {
            size_t ny = m_yorders[iy];
            ret.arr2d(ix, iy) = prod.block(x_offset, y_offset, nx, ny).sum();
            y_offset+=ny;
        }
        x_offset+=nx;
    }
}

void Integrator2Base::init_sampler() {
  auto trans=transformation_("points")
      .output("x")
      .output("y")
      .output("xedges")
      .output("yedges")
      .output("xmesh")
      .output("ymesh")
      .types(&Integrator2Base::check_sampler)
      .func(&Integrator2Base::sample)
      ;

  if(!m_xedges.size() && !m_yedges.size()){
    trans.input("edges") //hist with edges
         .types(TypesFunctions::if2d<0>, TypesFunctions::ifHist<0>);
  }
  else {
    trans.finalize();
  }
}

void Integrator2Base::check_sampler(TypesFunctionArgs& fargs){
  auto& rets=fargs.rets;
  rets[0] = DataType().points().shape(m_xweights.size());
  rets[1] = DataType().points().shape(m_yweights.size());

  rets[4] = DataType().points().shape(m_xweights.size(), m_yweights.size());
  rets[5] = DataType().points().shape(m_xweights.size(), m_yweights.size());

  auto& args=fargs.args;
  if(args.size() && !m_xedges.size()){
    auto& arg=args[0];
    auto& xedges=arg.edgesNd[0];
    m_xedges=Map<const ArrayXd>(xedges.data(), xedges.size());
    auto& yedges=arg.edgesNd[1];
    m_yedges=Map<const ArrayXd>(yedges.data(), yedges.size());
  }
  rets[2]=DataType().points().shape(m_xedges.size()).preallocated(m_xedges.data());
  rets[3]=DataType().points().shape(m_yedges.size()).preallocated(m_yedges.data());
}

void Integrator2Base::dump(){
    std::cout<<"X edges:   "<<m_xedges.transpose()<<std::endl;
    std::cout<<"X orders:  "<<m_xorders.transpose()<<std::endl;
    std::cout<<"X weights: "<<m_xweights.transpose()<<std::endl<<std::endl;

    std::cout<<"Y edges:   "<<m_yedges.transpose()<<std::endl;
    std::cout<<"Y orders:  "<<m_yorders.transpose()<<std::endl;
    std::cout<<"Y weights: "<<m_yweights.transpose()<<std::endl<<std::endl;
}