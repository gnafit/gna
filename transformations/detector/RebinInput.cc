#include "RebinInput.hh"
#include "TypeClasses.hh"
#include <algorithm>
#include <fmt/format.h>

//#define DEBUG_ENL
using GNA::DataPropagation;
using GNA::MatrixType;
using GNA::DataMutability;

#ifdef DEBUG_ENL
#define  DEBUG(...) do {                                      \
    printf(__VA_ARGS__);                                      \
    } while (0);
#else
#define  DEBUG(...)
#endif

RebinInput::RebinInput(int rounding, DataMutability input_edges_mode, DataPropagation propagate_matrix) :
HistSmearSparse(propagate_matrix, MatrixType::Any, "rebin", "histin", "histout"),
m_dynamic_edges(input_edges_mode==DataMutability::Dynamic),
m_round_scale{pow(10, rounding)}
{
    using namespace TypeClasses;

    transformation_("matrix")
        .input("HistEdgesOut", /*inactive*/true)
        .input("EdgesIn",      /*inactive*/input_edges_mode==DataMutability::Static)
        .output("FakeMatrix")
        .types(new CheckKindT<double>(DataKind::Hist, {0,0}), new CheckNdimT<double>(1, {0,1}))
        .types(&RebinInput::getEdges)
        .func(&RebinInput::calcMatrix);

    add_transformation();
    add_input();
    set_open_input();
}

void RebinInput::getEdges(TypesFunctionArgs& fargs) {
    auto& args = fargs.args;
    auto& dtype_out = args[0];
    if(dtype_out.defined() && m_edges_out.empty()){
        m_nbins_out = dtype_out.size();
        round(m_nbins_out+1u, dtype_out.edges.data(), m_edges_out);
    }

    auto& dtype_in = args[1];
    switch(dtype_in.kind){
        case DataKind::Hist:
            if(m_edges_in.empty()) {
                m_nbins_in = dtype_in.size();
                round(m_nbins_in+1u, dtype_in.edges.data(), m_edges_in);

                if(m_dynamic_edges){
                    throw args.error(dtype_in, fmt::format("Arg 1 should not be a Histogram. Dynamic input is expected."));
                }
            }
            break;
        case DataKind::Points:
            m_nbins_in = dtype_in.size();
            if(m_nbins_in>1){
                m_nbins_in--;
            }
            else{
                throw args.error(dtype_in, fmt::format("Arg 1 (points) should have at least 2 elements."));
            }
            if(!m_dynamic_edges){
                throw args.error(dtype_in, fmt::format("Arg 1 should not be a Points instance. Not dynamic input is expected."));
            }
            break;
        case DataKind::Undefined:
            break;
    }

    if(m_nbins_in>0u && m_nbins_out>0u){
        fargs.rets[0].points().shape(m_nbins_out, m_nbins_in);
    }
}

void RebinInput::round(size_t n, double const* edges_raw, std::vector<double>& edges_rounded){
    auto scale = m_round_scale;
    auto round_single = [scale](double num){ return std::round( num*scale )/scale; };
    edges_rounded.resize(n);
    std::transform(edges_raw, edges_raw+n, edges_rounded.begin(), round_single);
}

void RebinInput::calcMatrix(FunctionArgs& fargs) {
    auto& args=fargs.args;
    if(m_dynamic_edges && m_edges_in.empty()){
        auto& data_in = args[1];
        round(data_in.type.size(), data_in.buffer, m_edges_in);
    }

    size_t nbins_in=m_edges_in.size()-1;
    m_sparse_cache.resize(m_edges_out.size()-1, nbins_in);
    m_sparse_cache.reserve(nbins_in);
    m_sparse_cache.setZero();

    size_t inew{0};
    auto edge_new = m_edges_out.begin();
    auto edge_old = std::lower_bound(m_edges_in.begin(), m_edges_in.end(), *edge_new);
    size_t iold=std::distance(m_edges_in.begin(), edge_old);

    if (m_permit_underflow && *edge_new<m_edges_in.front()){
      edge_new = std::lower_bound(m_edges_out.begin(), m_edges_out.end(), *edge_old);
      inew = std::distance(m_edges_out.begin(), edge_new);
    }

    for (; inew < m_edges_out.size(); ++inew) {
        while(*edge_old<*edge_new) {
            m_sparse_cache.insert(inew-1, iold) = 1.0;

            ++edge_old;
            ++iold;
            if(edge_old==m_edges_in.end()){
                dump(m_edges_in.size(), m_edges_in.data(), m_edges_out.size(), m_edges_out.data());
                throw fargs.rets.error("RebinInput: bin edges are not consistent (outer)");
            }
        }
        if(*edge_new!=*edge_old){
            dump(m_edges_in.size(), m_edges_in.data(), m_edges_out.size(), m_edges_out.data());
            printf("Iteration %4zu-%4zu (new-old): %g-%g=%g\n", inew, iold, *edge_new, *edge_old, *edge_new-*edge_old);
            throw fargs.rets.error("RebinInput: bin edges are not consistent (inner)");
        }
        ++edge_new;
    }

    m_sparse_cache.makeCompressed();
    if ( m_propagate_matrix )
      fargs.rets[0].mat = m_sparse_cache;
}

void RebinInput::dump(size_t oldn, double* oldedges, size_t newn, double* newedges) const {
  std::cout<<"Rounding: "<<m_round_scale<<std::endl;
  std::cout<<"Old edges: " << Eigen::Map<const Eigen::Array<double,1,Eigen::Dynamic>>(oldedges, oldn)<<std::endl;
  std::cout<<"New edges: " << Eigen::Map<const Eigen::Array<double,1,Eigen::Dynamic>>(newedges, newn)<<std::endl;
}
