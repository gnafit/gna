#include "Integrator21Base.hh"

#include <iostream>
#include <iterator>

using namespace Eigen;
using namespace std;

Integrator21Base::Integrator21Base(size_t xbins, int xorders, double* xedges, int yorder, double ymin, double ymax) :
m_xorders(xbins),
m_yweights(yorder),
m_yorder(yorder),
m_ymin(ymin),
m_ymax(ymax),
{
    m_xorders.setConstant(xorders);
    init_base(xedges);
}

Integrator21Base::Integrator21Base(size_t xbins, int* xorders, double* xedges, int yorder, double ymin, double ymax) :
m_xorders(Map<const ArrayXi>(xorders, xbins)),
m_yweights(yorder),
m_yorder(yorder),
m_ymin(xmin),
m_ymax(xmax),
{
    init_base(xedges);
}

void Integrator21Base::init_base(double* xedges) {
    m_xweights.resize(m_orders.sum());

    transformation_("hist")
        .input("f")
        .output("hist")
        .types(TypesFunctions::ifPoints<0>, &Integrator21Base::check_base)
        .func(&Integrator21Base::integrate)
        ;

    if(xedges){
        m_xedges=Map<const ArrayXd>(xedges, m_xorders.size()+1);
    }
}

void Integrator21Base::check_base(TypesFunctionArgs& fargs){
    //fargs.rets[0]=DataType().hist().edges(m_edges.size(), m_edges.data());

    //if (fargs.args[0].shape[0] != m_weights.size()){
        //throw fargs.args.error(fargs.args[0], "inconsistent function size");
    //}

    //if((m_orders<1).any()){
        //std::cerr<<m_orders<<std::endl;
        //throw std::runtime_error("All integration orders should be >=1");
    //}
}

void Integrator21Base::integrate(FunctionArgs& fargs){
    //auto* ret=fargs.rets[0].buffer;
    //auto& fun=fargs.args[0].x;

    //ArrayXd prod = fargs.args[0].x*m_weights;
    //auto* data = prod.data();
    //auto* order = m_orders.data();
    //for (size_t i = 0; i < m_orders.size(); ++i) {
        //auto* data_next=std::next(data, *order+m_shared_edge);
        //*ret = std::accumulate(data, data_next, 0.0);
        //if(m_shared_edge){
            //data=prev(data_next);
        //}
        //else{
            //data=data_next;
        //}
        //advance(order,1);
        //advance(ret,1);
    //}
}

void Integrator21Base::init_sampler() {
  //auto trans=transformation_("points")
      //.output("x")
      //.output("xedges")
      //.types(&Integrator21Base::check_sampler)
      //.func(&Integrator21Base::sample)
      //;

  //if(m_edges.size()){
    //trans.finalize();
  //}
  //else{
    //trans.input("edges") //hist with edges
      //.types(TypesFunctions::if1d<0>, TypesFunctions::ifHist<0>, TypesFunctions::binsToEdges<0,1>);
  //}
}

void Integrator21Base::check_sampler(TypesFunctionArgs& fargs){
  //auto& rets=fargs.rets;
  //rets[0] = DataType().points().shape(m_weights.size());

  //auto& args=fargs.args;
  //if(args.size() && !m_edges.size()){
    //auto& edges=fargs.args[0].edges;
    //m_edges=Map<const ArrayXd>(edges.data(), edges.size());
  //}
  //rets[1]=DataType().points().shape(m_edges.size()).preallocated(m_edges.data());
}
void Integrator21Base::dump(){
    //std::cout<<"Edges: "<<m_edges.transpose()<<std::endl;
    //std::cout<<"Orders: "<<m_orders.transpose()<<std::endl;
    //std::cout<<"Weights: "<<m_weights.transpose()<<std::endl;
}
