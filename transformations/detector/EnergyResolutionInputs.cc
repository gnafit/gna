#include <boost/math/constants/constants.hpp>
#include "EnergyResolutionInputs.hh"
#include "TypeClasses.hh"
#include <fmt/format.h>
#include <string.h>

// #define DEBUG_ERES_INPUTS
#ifdef DEBUG_ERES_INPUTS
    #include <iostream>
    using std::cout;
    using std::endl;
    #define DEBUG_START_IN  5040
    #define DEBUG_START_OUT 2280
    #define DEBUG_LEN 20
#endif

constexpr double one_div_root_two_pi = boost::math::constants::one_div_root_two_pi<double>();
constexpr double root_two_pi = boost::math::constants::root_two_pi<double>();
constexpr double half = boost::math::constants::half<double>();

using GNA::DataMutability;
using GNA::DataPropagation;
using GNA::MatrixType;

EnergyResolutionInputs::EnergyResolutionInputs(DataMutability input_edges_mode) :
HistSmearSparse(DataPropagation::Propagate, MatrixType::Any),
m_dynamic_edges(input_edges_mode==DataMutability::Dynamic)
{
    using namespace TypeClasses;

    this->transformation_("matrix")
        .input("HistEdgesOut", /*inactive*/true)                                // Output bin edges [N]
        .input("EdgesIn", /*inactive*/input_edges_mode==DataMutability::Static) // Input bin edges [M]
        .input("RelSigma")                                                      // Relative Sigma value for each bin center [M-1]
        .output("FakeMatrix")
        .types(new CheckKindT<double>(DataKind::Hist, 0), new CheckKindT<double>(DataKind::Points, 2))
        .types(new CheckNdimT<double>(1))
        .types(&EnergyResolutionInputs::types)
        .func(&EnergyResolutionInputs::calcMatrix);

    add_transformation();
    add_input();
    set_open_input();

    m_delta_threshold = sqrt(-2.0*log(root_two_pi*m_cell_threshold));
}

double EnergyResolutionInputs::resolution(double relDelta, double Sigma) const noexcept {
  return one_div_root_two_pi*std::exp(-0.5*pow(relDelta, 2))/Sigma;
}

void EnergyResolutionInputs::calcMatrix(FunctionArgs& fargs) {
    auto& args = fargs.args;

    if(m_dynamic_edges && m_edges_in.size()==0u){
        m_edges_in = args[1].x;
        m_nbins_in = m_edges_in.size()-1;
        processEdgesIn();
    }

    m_matrix.setZero();
    m_matrix.resize(m_nbins_out, m_nbins_in);

    auto& relsigmas = args[2].x;
    auto relsigma_fcn = relsigmas.size()==static_cast<int>(m_nbins_in) ?
                           std::function{[&relsigmas](size_t idx){ return relsigmas(idx); }} :
                           std::function{[&relsigmas](size_t idx){ return 0.5*(relsigmas(idx)+relsigmas(idx+1)); }};

    /* fill the cache matrix with probalilities for number of events to leak to other bins */
    /* colums corressponds to reconstrucred energy and rows to true energy */
    for (size_t idx_in = 0; idx_in < m_nbins_in; ++idx_in) {
        auto Etrue    = m_centers_in(idx_in);
        auto relsigma = relsigma_fcn(idx_in);

        double colsum=0.0;
        for (size_t idx_out = 0; idx_out < m_nbins_out; ++idx_out) {
            auto Erec  = m_centers_out(idx_out);
            auto dErec = m_widths_out(idx_out);

            auto sigma = Etrue*relsigma;
            auto relDelta = (Etrue-Erec)/sigma;
            // if (relDelta>m_delta_threshold){
            //     continue;
            // }
            auto rEvents = dErec*resolution(relDelta, sigma);
            if (rEvents<m_cell_threshold){
                continue;
            }

            colsum+=rEvents;
            m_matrix(idx_out, idx_in) = rEvents;
        }

        if(m_matrix(0, idx_in)==0.0 && m_matrix(m_nbins_out-1, idx_in)==0.0){
            if(colsum>0.0){
              m_matrix.col(idx_in)/=colsum;
            }
        }
    }

    m_sparse_cache.resize(m_nbins_out, m_nbins_in);
    m_sparse_cache.setZero();
    m_sparse_cache = m_matrix.matrix().sparseView();
    m_sparse_cache.makeCompressed();
}

void EnergyResolutionInputs::types(TypesFunctionArgs& fargs) {
    auto& args = fargs.args;

    auto& dtype_out = args[0];
    if(dtype_out.defined() && m_edges_out.size()==0u){
        m_nbins_out = dtype_out.size();
        m_edges_out = Eigen::Map<const Eigen::ArrayXd>(dtype_out.edges.data(), dtype_out.edges.size());
        processEdgesOut();
    }

    auto& dtype_in = args[1];
    switch(dtype_in.kind){
        case DataKind::Hist:
            if(m_edges_in.size()==0u) {
                m_nbins_in = dtype_in.size();
                m_edges_in = Eigen::Map<const Eigen::ArrayXd>(dtype_in.edges.data(), dtype_in.edges.size());
                processEdgesIn();

                if(m_dynamic_edges){
                    throw args.error(dtype_in, fmt::format("Arg 1 (EdgesIn) should not be a Histogram. Dynamic input is expected."));
                }
            }
            break;

        case DataKind::Points:
            m_nbins_in = dtype_in.size();
            if(m_nbins_in>1){
                m_nbins_in--;
            }
            else{
                throw args.error(dtype_in, fmt::format("Arg 1 (EdgesIn, points) should have at least 2 elements."));
            }
            if(!m_dynamic_edges){
                throw args.error(dtype_in, fmt::format("Arg 1 (EdgesIn) should not be a Points instance. Not dynamic input is expected."));
            }
            break;

        case DataKind::Undefined:
            break;
    }

    if(m_nbins_in>0u && m_nbins_out>0u){
        auto& ret = fargs.rets[0];
        m_matrix.resize(m_nbins_out, m_nbins_in);
        ret.points().shape(m_nbins_out, m_nbins_in).preallocated(m_matrix.data());
    }

    auto& dtype_relsigma = args[2];
    if(m_nbins_in>0u && dtype_relsigma.defined()){
        auto n_rs = dtype_relsigma.size();
        if (n_rs!=m_nbins_in && n_rs!=(m_nbins_in+1)){
            throw args.error(dtype_relsigma, fmt::format("Arg 2 (RelSigma) should be of size of N bins (in) or N edges (in)."));
        }
    }
}

void EnergyResolutionInputs::processEdgesOut(){
    m_centers_out = half*(m_edges_out.tail(m_nbins_out) + m_edges_out.head(m_nbins_out));
    m_widths_out = m_edges_out.tail(m_nbins_out) - m_edges_out.head(m_nbins_out);
}

void EnergyResolutionInputs::processEdgesIn(){
    m_centers_in = half*(m_edges_in.tail(m_nbins_in) + m_edges_in.head(m_nbins_in));

#ifdef DEBUG_ERES_INPUTS
    cout<<"Edges in: "<<m_nbins_in+1u<<"\n"<<m_edges_in.transpose().segment(DEBUG_START_IN, DEBUG_LEN+1)<<endl;
    cout<<"Edges out: "<<m_nbins_out+1u<<"\n"<<m_edges_out.transpose().segment(DEBUG_START_OUT, DEBUG_LEN+1)<<endl;
    cout<<"Centers (in): "<<m_centers_in.rows()<<" "<<m_centers_in.cols()<<"\n"<<m_centers_in.transpose().segment(DEBUG_START_IN, DEBUG_LEN)<<endl;
    cout<<"Centers (out): "<<m_centers_out.rows()<<" "<<m_centers_out.cols()<<"\n"<<m_centers_out.transpose().segment(DEBUG_START_OUT, DEBUG_LEN)<<endl;
    cout<<"Widths (out): "<<m_widths_out.rows()<<" "<<m_widths_out.cols()<<"\n"<<m_widths_out.transpose().segment(DEBUG_START_OUT, DEBUG_LEN)<<endl;
#endif
}

