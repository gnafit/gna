#include "IntegratorBase.hh"
#include "fmt/format.h"

#include <string>
#include <iostream>
#include <iterator>

using namespace Eigen;
using namespace std;

IntegratorBase::IntegratorBase(int order, bool shared_edge) :
GNAObjectBind1N<double>("hist", "f", "hist", 1, 0, 0),
m_order(static_cast<size_t>(order)),
m_shared_edge(static_cast<size_t>(shared_edge))
{
}

IntegratorBase::IntegratorBase(size_t bins, int orders, double* edges, bool shared_edge) :
GNAObjectBind1N<double>("hist", "f", "hist", 1, 0, 0),
m_orders(bins),
m_shared_edge(static_cast<size_t>(shared_edge))
{
    m_orders.setConstant(orders);
    init_base(edges);
}

IntegratorBase::IntegratorBase(size_t bins, int *orders, double* edges, bool shared_edge) :
GNAObjectBind1N<double>("hist", "f", "hist", 1, 0, 0),
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
}

TransformationDescriptor IntegratorBase::add_transformation(const std::string& name){
    this->transformation_(this->new_transformation_name(name))
        .types(TypesFunctions::if1d<0>, TypesFunctions::ifPoints<0>, &IntegratorBase::check_base)
        .types(TypesFunctions::ifSame)
        .func(&IntegratorBase::integrate)
        ;
    reset_open_input();
    return transformations.back();
}

void IntegratorBase::check_base(TypesFunctionArgs& fargs){
    auto& rets=fargs.rets;
    for(size_t i(0); i<rets.size(); ++i)
    {
        rets[i]=DataType().hist().edges(m_edges.size(), m_edges.data());;
    }

    if (fargs.args[0].shape[0] != static_cast<size_t>(m_weights.size())){
        throw fargs.args.error(fargs.args[0], "inconsistent function size");
    }

    if((m_orders<(2*int(m_shared_edge))).any()){
        std::cerr<<m_orders<<std::endl;
        if(m_shared_edge){
          throw std::runtime_error("All integration orders should be >=2");
        }
        else{
          throw std::runtime_error("All integration orders should be >=0");
        }
    }
}

void IntegratorBase::integrate(FunctionArgs& fargs){
    auto& args=fargs.args;
    auto& rets=fargs.rets;
    for (size_t i = 0; i < args.size(); ++i) {
        auto* ret=rets[i].buffer;

        ArrayXd prod = args[i].x*m_weights;
        auto* data = prod.data();
        auto* order = m_orders.data();
        for (int i = 0; i < m_orders.size(); ++i) {
            if(*order){
                auto* data_next=std::next(data, *order);
                *ret = std::accumulate(data, data_next, 0.0);
                if(m_shared_edge){
                    data=prev(data_next);
                }
                else{
                    data=data_next;
                }
            }
            else{
                *ret = 0.0;
            }
            advance(order,1);
            advance(ret,1);
        }
    }
}

void IntegratorBase::init_sampler() {
    auto trans=transformation_("points")
        .output("x")        // 0
        .output("xedges")   // 1
        .output("xhist")    // 2
        .output("xcenters") // 3
        .types(&IntegratorBase::check_sampler)
        .func(&IntegratorBase::sample)
        ;

    if(!m_edges.size()){
        trans.input("edges", /*inactive*/true) //hist with edges
            .types(TypesFunctions::if1d<0>, TypesFunctions::ifHist<0>);
    }
    trans.finalize();

    add_transformation();
    add_input();
    set_open_input();
}

void IntegratorBase::check_sampler(TypesFunctionArgs& fargs){
    auto& rets=fargs.rets;
    auto& args=fargs.args;

    if(args.size() && !m_edges.size()){
        auto& arg=fargs.args[0];
        if(arg.kind==DataKind::Undefined){
            return;
        }

        auto& edges=arg.edges;
        m_edges=Map<const ArrayXd>(edges.data(), edges.size());

        if(m_order){
            m_orders.resize(arg.size());
            m_orders.setConstant(m_order.value());
            m_order.reset();
            init_base(nullptr);
        }
    }

    /*x*/        rets[0].points().shape(m_weights.size());
    /*xedges*/   rets[1].points().shape(m_edges.size()).preallocated(m_edges.data());
    /*xhist*/    rets[2].hist().edges(m_edges.size(), m_edges.data());;
    /*xcenters*/ rets[3].points().shape(m_edges.size()-1);
}

void IntegratorBase::dump(){
    std::cout<<"Edges: "<<m_edges.transpose()<<std::endl;
    std::cout<<"Orders: "<<m_orders.transpose()<<std::endl;
    std::cout<<"Weights: "<<m_weights.transpose()<<std::endl;
}

void IntegratorBase::set_edges(OutputDescriptor& hist_output){
    auto& inputs=transformations.front().inputs;
    if(!inputs.size()){
        throw std::runtime_error("Can not bind bin edges since bin edges were provided in the constructor.");
    }
    inputs.front()(hist_output);
}
