#include "SamplerBase.hh"

#include <iostream>
#include <iterator>

using namespace Eigen;
using namespace std;

SamplerBase::SamplerBase(size_t bins, int orders, double* edges) :
m_orders(bins)
{
    m_orders.setConstant(orders);
    init_base(edges);
}

SamplerBase::SamplerBase(size_t bins, int *orders, double* edges) :
m_orders(Map<const ArrayXi>(orders, bins))
{
    init_base(edges);
}

void SamplerBase::init_base(double* edges) {
    m_weights.resize(m_orders.sum());

    transformation_("hist")
        .input("f")
        .output("hist")
        .types(TypesFunctions::if1d<0>, TypesFunctions::ifPoints<0>, &SamplerBase::check_base)
        .func(&SamplerBase::integrate)
        ;

    if(edges){
        m_edges=Map<const ArrayXd>(edges, m_orders.size()+1);
    }
}

void SamplerBase::set_shared_edge(){
    if(m_shared_edge){
        throw std::runtime_error("may not set shared edge twice");
    }
    m_shared_edge=1;
    m_weights.resize(m_weights.size()+1);
}

void SamplerBase::check_base(TypesFunctionArgs& fargs){
    fargs.rets[0]=DataType().hist().edges(m_edges.size(), m_edges.data());

    if (fargs.args[0].shape[0] != m_weights.size()){
        throw fargs.args.error(fargs.args[0], "inconsistent function size");
    }

    if((m_orders<1).any()){
        std::cerr<<m_orders<<std::endl;
        throw std::runtime_error("All integration orders should be >=1");
    }
}

void SamplerBase::integrate(FunctionArgs& fargs){
    auto* ret=fargs.rets[0].buffer;
    auto& fun=fargs.args[0].x;

    ArrayXd prod = fargs.args[0].x*m_weights;
    auto* data = prod.data();
    auto* order = m_orders.data();
    for (size_t i = 0; i < m_orders.size(); ++i) {
        auto* data_next=std::next(data, *order+m_shared_edge);
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

void SamplerBase::dump(){
    std::cout<<"Edges: "<<m_edges.transpose()<<std::endl;
    std::cout<<"Orders: "<<m_orders.transpose()<<std::endl;
    std::cout<<"Weights: "<<m_weights.transpose()<<std::endl;
}
